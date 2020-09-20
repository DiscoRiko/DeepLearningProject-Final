import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from .cs236781.train_results import BatchResult, EpochResult, FitResult


class Trainer(abc.ABC):
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = \
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            actual_num_epochs += 1

            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            tr_er_loss, tr_er_acc = train_result
            train_loss.append(sum(tr_er_loss) / len(tr_er_loss))
            train_acc.append(tr_er_acc)

            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            te_er_loss, te_er_acc = test_result
            test_loss.append(sum(te_er_loss) / len(te_er_loss))
            test_acc.append(te_er_acc)

            test_loss_len = len(test_loss)
            if test_loss_len > 1:
                if test_loss[test_loss_len - 1] < test_loss[test_loss_len - 2]:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping:
                break
            
            test_acc_len = len(test_acc)
            if actual_num_epochs == 1:
              best_acc = test_acc[test_acc_len - 1]
              save_checkpoint = True
            else:
              if(best_acc < test_acc[test_acc_len - 1]):
                best_acc = test_acc[test_acc_len - 1]
                save_checkpoint = True
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        batch_acc = 0
        #num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                #num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                batch_acc += batch_res.batch_acc

            avg_loss = sum(losses) / num_batches
            #print(f"num_correct after epoch:{num_correct}")
            #print(f"num_samples after epoch:{num_samples}")
            #accuracy = 100. * num_correct / num_samples
            accuracy = 100. * batch_acc / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class CenterFaceMaskTrainer(Trainer):
    def __init__(self, model, loss_fn, lambdas, optimizer, device=None):
        super().__init__(model, loss_fn, lambdas, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        images, indicators, actual_sizes, actual_center_points, actual_final_masks = batch['image'], batch['indicators'], batch['sizes'], batch['centers'], batch['masks']
        images = images.to(self.device, dtype=torch.float)  # (N,3,H,W)
        indicators = indicators.to(self.device, dtype=torch.float)  # (N,1,C)
        #classes = indicators.shape[2]

        # TODO:
        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate number of correct char predictions
        # ====== YOUR CODE: ======
        sizes, center_points, final_masks = self.model(images)

        loss = self.loss_fn(sizes, actual_sizes, center_points, actual_center_points, final_masks, actual_final_masks, self.lambdas)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        batch_acc = get_acc(final_masks, actual_final_masks)
        # ========================

        return BatchResult(loss.item(), batch_acc)

    def test_batch(self, batch) -> BatchResult:
        images, indicators, actual_sizes, actual_center_points, actual_final_masks = batch['image'], batch['indicators'], batch['sizes'], batch['centers'], batch['masks']
        images = images.to(self.device, dtype=torch.float)  # (N,3,H,W)
        indicators = indicators.to(self.device, dtype=torch.float)  # (N,1,C)
        #classes = indicators.shape[2]

        with torch.no_grad():
            # TODO:
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            sizes, center_points, final_masks = self.model(images)

            loss = self.loss_fn(sizes, actual_sizes, center_points, actual_center_points, final_masks, actual_final_masks, self.lambdas)

            batch_acc = get_acc(final_masks, actual_final_masks)
            # ========================
        return BatchResult(loss.item(), batch_acc)


def get_acc(final_masks, actual_final_masks):
    """
    final masks - list of N dictionaries: {organ: (final_mask, indices_for_cropping)}
    actual_final_masks - list of N dictionaries: {organ: (actual_final_mask)}
    """
    final_masks_list = []
    indices_for_cropping_list = []
    actual_final_masks_list = []
    full_final_masks_list = []

    # extraction final mask and indices from the dict
    for final_mask_dict in final_masks:
        for organ, data in final_mask_dict.items():
            final_mask, indices_for_cropping = data
            final_masks_list.append(final_mask)
            indices_for_cropping_list.append(indices_for_cropping)

    # extraction actual final mask from the dict
    for actual_final_masks_dict in actual_final_masks:
        for organ, data in actual_final_masks_dict.items():
            actual_final_mask = data
            actual_final_masks_list.append(actual_final_mask)

    # crop the actual final masks
    for indices_for_cropping, final_mask in zip(indices_for_cropping_list, final_masks_list):
        full_final_mask = torch.zeros((512, 512))
        r1, r2, c1, c2 = indices_for_cropping
        full_final_mask[r1:r2, c1:c2] = final_mask
        full_final_masks_list.append(full_final_mask)

    actual_final_masks = torch.stack(actual_final_masks_list)
    full_final_masks = torch.stack(full_final_masks_list)
    sig = torch.nn.Sigmoid()
    full_final_masks = sig(full_final_masks)
    full_final_masks = full_final_masks > 0.9
    full_final_masks = full_final_masks.to(dtype=torch.float32)

    acc = (actual_final_masks == full_final_masks).to(dtype=torch.float32).mean()
    return acc
