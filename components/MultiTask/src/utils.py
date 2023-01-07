# Requirements
import numpy as np
import torch
import random
import os

# Set seed
def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Method to Convert to ONNX 
def convert_ONNX(max_length:int, model:torch.nn.Module, output_path:str): 
    # set the model to inference mode 
    model.to(torch.device('cpu'))
    model.eval()

    # Let's create a dummy input tensor
    dummy_input_1 = torch.randint(0, 2, (1, max_length))
    dummy_input_2 = torch.randint(0, 2, (1, max_length))

    # Export the model   
    torch.onnx.export(model,                            # model being run 
         (dummy_input_1, dummy_input_2),                # model input (or a tuple for multiple inputs) 
         output_path,                                   # where to save the model  
         export_params=True,                            # store the trained parameter weights inside the model file 
         opset_version=14,                              # the ONNX version to export the model to 
         do_constant_folding=True,                      # whether to execute constant folding for optimization 
         input_names = ['input_ids', 'attention_mask'], # the model's input names 
         output_names = ['ic_output', 'ner_output'],    # the model's output names 
         operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, # https://github.com/facebookresearch/fairseq/issues/3395
         dynamic_axes={                                 # variable length axes 
             'input_ids' : {0 : 'batch_size'},    
             'attention_mask' : {0 : 'batch_size'},
             'ic_output' : {0 : 'batch_size'},
			 'ner_output' : {0 : 'batch_size'},
             })

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