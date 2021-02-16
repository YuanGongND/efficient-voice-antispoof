import logging
from nnet.metrics import compute_eer
from nnet.validation import compute_loss, utt_scores


# Get the same logger from main"
logger = logging.getLogger("MLP")


def prediction(args, model, device, eval_loader, eval_utt2label):
    logger.info("Starting evaluation")
    eval_loss, eval_scores = compute_loss(model, device, eval_loader)
    eval_preds, eval_labels = utt_scores(eval_scores, eval_utt2label)
    eval_eer = compute_eer(eval_labels, eval_preds)

    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                eval_loss, eval_eer))

    return eval_loss, eval_eer

