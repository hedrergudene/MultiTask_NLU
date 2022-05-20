# Requirements
import numpy as np
import torch

# Helper method to collate utterances
def collate_spaCy_HuggingFace(text, nlp, tokenizer, MAX_LEN, ner2idx, label):
    # Build dictionaries
    doc = nlp(text)
    entlist = [(elem.label_, elem.start_char, elem.end_char) for elem in doc.ents if elem.label_ in label]
    # Tokenize text
    tokens = tokenizer.encode_plus(text, max_length=MAX_LEN, padding='max_length',
                                    truncation=True, return_offsets_mapping=True)
    train_tokens = tokens['input_ids']
    attn_tokens = tokens['attention_mask']

    # Create array to store each class labels
    ## First axis indicates the label
    ## Second axis each text
    ## Third axis the token position
    targets = np.zeros((MAX_LEN), dtype='int32') #Everything is unlabelled by Default

    # FIND TARGETS IN TEXT AND SAVE IN TARGET ARRAYS
    offsets = tokens['offset_mapping']
    offset_index = 0
    for index, (label, start, end) in enumerate(entlist):
        a = int(start)
        b = int(end)
        if offset_index>len(offsets)-1:
            break
        c = offsets[offset_index][0] # Token start
        d = offsets[offset_index][1] # Token end
        count_token = 0 # token counter
        beginning = True
        while b>c: # While tokens lie in the discourse of a specific entity
            if (c>=a)&(b>=d): # If token is inside discourse
                if beginning:
                    targets[offset_index] = ner2idx['B-'+label]
                    beginning = False
                else:
                    targets[offset_index] = ner2idx['I-'+label]
            count_token += 1
            offset_index += 1 # Move to the next token
            if offset_index>len(offsets)-1: # If next token is out of this entity range, jump to the next row of the df
                break
            c = offsets[offset_index][0]
            d = offsets[offset_index][1]
    # 'PAD' label to make loss function ignore padding, which is basically where attn_tokens is zero
    targets[np.where(np.array(attn_tokens)==0)[0]] = ner2idx['PAD']
    # Save in dictionary
    target =  torch.LongTensor(targets)
    # End of method
    return {"tokens":torch.LongTensor(train_tokens), "attn_mask":torch.LongTensor(attn_tokens)}, target


# Method to ensemble NER outputs into one
def convert_tags(tag, original_idxs):
    tag1, tag2= tag.split('.')[0], tag.split('.')[1]
    # EXCLUSIVE CONDITIONS
    ## Check if the first of them is 'O'
    if tag1=='O':
        return 'O'
    ## Both will agree on pads so does not matter which to choose
    elif (tag1=='PAD') or (tag2=='PAD'):
        return 'PAD'
    ## Check if both agree in the same scheme word, otherwise return 'O'
    ## (first condition is redundant given previous ones but we keep for clarity)
    elif (tag1[0] in ['I','B']) and (tag2[0] in ['I','B']) and (tag1[0]!=tag2[0]):
        return 'O'
    ## At this point, first tag is something 'relevant' and alligned with the 
    ## second one, possibly being the latter 'O'. So now we start with...
    
    # INCLUSIVE CONDITIONS
    ## Check if second one is 'O' and first one is in original dict
    elif (tag2=='O') and (tag1 in list(original_idxs.values())):
        return tag1
    ## Check if second one is not 'O' and smart concatenation of them is in
    ## original dictionary; i.e., remove scheme from second (they already agree)
    elif (tag2!='O') and (tag1+'.'+tag2[2:] in list(original_idxs.values())):
        return tag1+tag2[2:]
    
    # ESCAPE CONDITION
    ## The last use case is that the concatenation of them do not lie in the
    ## original dictionary, so we assign 'O'
    else:
        return 'O'


# Helper methods to compute F1-score in NER
def startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
      chunkStart = True
    if prevTag == 'I' and tag == 'B':
      chunkStart = True
    if prevTag == 'O' and tag == 'B':
      chunkStart = True
    if prevTag == 'O' and tag == 'I':
      chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
      chunkStart = True
    return chunkStart


def endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
      chunkEnd = True
    if prevTag == 'B' and tag == 'O':
      chunkEnd = True
    if prevTag == 'I' and tag == 'B':
      chunkEnd = True
    if prevTag == 'I' and tag == 'O':
      chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
      chunkEnd = True
    return chunkEnd


def splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
      raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
      tag = s[0]
      tagType = ""
    else:
      tag = s[0]
      tagType = s[1]
    return tag, tagType


def computeF1Score(input, target):
	correctChunk = {}
	correctChunkCnt = 0.0
	foundCorrect = {}
	foundCorrectCnt = 0.0
	foundPred = {}
	foundPredCnt = 0.0
	correctTags = 0.0
	tokenCount = 0.0
	for pred_slot, correct_slot in zip(input, target):
		inCorrect = False
		lastCorrectTag = 'O'
		lastCorrectType = ''
		lastPredTag = 'O'
		lastPredType = ''
		for p, c in zip(pred_slot, correct_slot):
			correctTag, correctType = splitTagType(c)
			predTag, predType = splitTagType(p)

			if inCorrect == True:
				if endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
					endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
					(lastCorrectType == lastPredType):
					inCorrect = False
					correctChunkCnt += 1.0
					if lastCorrectType in correctChunk:
						correctChunk[lastCorrectType] += 1.0
					else:
						correctChunk[lastCorrectType] = 1.0
				elif endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
					endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
					(correctType != predType):
					inCorrect = False

			if startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
				startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
				(correctType == predType):
				inCorrect = True

			if startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
				foundCorrectCnt += 1
				if correctType in foundCorrect:
					foundCorrect[correctType] += 1.0
				else:
					foundCorrect[correctType] = 1.0

			if startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
				foundPredCnt += 1.0
				if predType in foundPred:
					foundPred[predType] += 1.0
				else:
					foundPred[predType] = 1.0

			if correctTag == predTag and correctType == predType:
				correctTags += 1.0

			tokenCount += 1.0

			lastCorrectTag = correctTag
			lastCorrectType = correctType
			lastPredTag = predTag
			lastPredType = predType

		if inCorrect == True:
			correctChunkCnt += 1.0
			if lastCorrectType in correctChunk:
				correctChunk[lastCorrectType] += 1.0
			else:
				correctChunk[lastCorrectType] = 1.0

	if foundPredCnt > 0:
		precision = 1.0 * correctChunkCnt / foundPredCnt
	else:
		precision = 0

	if foundCorrectCnt > 0:
		recall = 1.0 * correctChunkCnt / foundCorrectCnt
	else:
		recall = 0

	if (precision + recall) > 0:
		f1 = (2.0 * precision * recall) / (precision + recall)
	else:
		f1 = 0

	return f1, precision, recall
