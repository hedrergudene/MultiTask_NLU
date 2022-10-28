# Requirements
from typing import List, Dict, Optional
import torch

# Focal Loss
class FocalLoss(torch.torch.nn.Module):
    """
    Implementation of FocalLoss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 gamma:float=1.,
                 temperature:float=1.,
                 from_logits:bool = True,
                 multilabel:bool=False,
                 reduction:str = 'mean',
                 n_classes:int = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:
        """
        Args:
        """
        super(FocalLoss, self).__init__()
        # Validations
        if not torch.is_tensor(class_weights) and (class_weights is not None):
            raise TypeError("Class weights type is not a torch.Tensor. Got {}"
                            .format(type(class_weights)))
        if class_weights is not None:
            if len(class_weights.shape)!=1:
                raise TypeError("Class weights do not have the right shape. Got shape {}"
                                .format(len(class_weights.shape)))
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction should be one of these values: {}"
                            .format(', '.join(['mean', 'sum', 'none'])))
        # Loss config settings
        self.from_logits=from_logits
        self.multilabel = multilabel
        self.reduction=reduction
        # Loss parameters
        self.gamma = gamma
        self.temperature = temperature
        self.class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.unsqueeze(dim=-1).to(device)
        self.n_classes = n_classes
        self.eps = 1e-6

    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. The last dimension must contain the proability distribution for each
                                  of the classes; i.e., input shape=(batch_size, n_classes) for IntentClassification problems, and
                                  input shape=(batch_size, max_postion_embeddings, n_classes) for NER problems.
            target (torch.Tensor): Batch containing ground truth, either in shape of binarised or one-hot encoded labels.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        # Part I: Validations
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        # Part II: Labels preprocessing
        # One-hot encode labels
        if len(target.shape) < len(input.shape): target = torch.torch.nn.functional.one_hot(target, num_classes=self.n_classes).float()

        # Part III: Compute loss
        loss = self.compute_loss(input, target)

        # Part IV:  Apply reduction method
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else: # Sum is already done in class weighting
            #loss = torch.sum(loss, dim = -1)
            return loss

    def compute_loss(self,
                     input:torch.Tensor,
                     target:torch.Tensor,
                     )->torch.Tensor:
        if self.from_logits:
            input_norm = torch.torch.nn.functional.logsigmoid(input) if self.multilabel else torch.torch.nn.functional.log_softmax(input/self.temperature, dim=-1)
            input_soft = torch.sigmoid(input)+self.eps if self.multilabel else torch.torch.nn.functional.softmax(input/self.temperature, dim=-1)+self.eps
        else:
            input_norm = torch.log(input)
            input_soft = input+self.eps
        # Compute the actual focal loss and weights classes
        focal_weight = torch.pow(1. - input_soft, self.gamma)
        focal_weights = - target * focal_weight * input_norm
        if len(focal_weights.shape)<3:
            focal_weights = torch.unsqueeze(focal_weights, dim=1)
        focal_loss = torch.bmm(focal_weights, self.class_weights.repeat(input.shape[0],1,1))
        return torch.squeeze(focal_loss)


# Conditional Random Field
class CRF(torch.nn.Module):
    """Conditional random field adapted from
        https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html#CRF
       to take into account GPU hardware acceleration.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
        device: Hardware in which algorithm is to be trained on.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmatorch.nn. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self,
                 num_tags: int,
                 batch_first: bool = True,
                 device: str = 'cpu',
                 ) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = torch.nn.Parameter(torch.empty(num_tags)).to(device)
        self.end_transitions = torch.nn.Parameter(torch.empty(num_tags)).to(device)
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags)).to(device)
        self.device = device

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        torch.nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.transitions, -0.1, 0.1)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()


    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)


    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score = score + emissions[0, torch.arange(batch_size), tags[0]] # Remove inplace ops

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score = score + self.transitions[tags[i - 1], tags[i]] * mask[i] # Remove inplace ops

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score = score + emissions[i, torch.arange(batch_size), tags[i]] * mask[i] # Remove inplace ops

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score = score + self.end_transitions[last_tags] # Remove inplace ops

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score = score + self.end_transitions # Remove inplace ops

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score = score + self.end_transitions # Remove inplace ops

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


# MultiTask loss function
class IC_NER_Loss(torch.nn.Module):
    """
    Implementation of IC_NER_Loss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 loss_type:str,
                 gamma:float=1.,
                 temperature:float=1.,
                 label_smoothing:float=.1,
                 from_logits:bool = True,
                 multilabel:bool= True,
                 reduction:str = 'mean',
                 n_classes:Dict = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:
        """
        Args:
        """
        super(IC_NER_Loss, self).__init__()
        # Validations
        assert loss_type in ['CELoss', 'FocalLoss', 'Mixed'], f"Loss type should either be 'CELoss', 'FocalLoss' or 'Mixed'."
        # Loss config settings
        self.alpha = torch.nn.Parameter(torch.zeros((1)), requires_grad=True).to(device)
        if loss_type=='CELoss':
            self.loss_fn=torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
        elif loss_type=='FocalLoss':
            self.loss_ic=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['IC'], class_weights, device)
            self.loss_ner=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['NER'], class_weights, device)
        else:
            self.loss_ic=torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
            self.loss_ner=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['NER'], class_weights, device)
        # Parameters
        self.loss_type = loss_type
                    

    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. The last dimension must contain the proability distribution for each
                                  of the classes; i.e., input shape=(batch_size, n_classes) for IntentClassification problems, and
                                  input shape=(batch_size, max_postion_embeddings, n_classes) for NER problems.
            target (torch.Tensor): Batch containing ground truth, either in shape of binarised or one-hot encoded labels.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        if self.loss_type=='CELoss':
            ic_loss = self.loss_fn(input[0], target['IC'])
            ner_loss = self.loss_fn(torch.permute(input[1], (0,2,1)), target['NER'])
        else:
            ic_loss = self.loss_ic(input[0], target['IC'])
            ner_loss = self.loss_ner(input[1], target['NER'])
        summary_loss = torch.sigmoid(self.alpha)*ic_loss + (1-torch.sigmoid(self.alpha))*ner_loss
        return {'IC':ic_loss, 'NER':ner_loss, 'summary':summary_loss}