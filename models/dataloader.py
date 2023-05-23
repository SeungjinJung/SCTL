import pickle
import torch

def dataloader(file, args):
    with open(file, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)
        
    if args.logger != 'None':
        args.logger.info("{}".format(file.split('/')[-1].split('.p')[0]))
        args.logger.info("y_probs_val : {} | y_val : {} | y_probs_test : {} | y_true_test : {}".format(y_probs_val.shape,y_val.shape,y_probs_test.shape,y_test.shape))

    n_class = y_probs_val.shape[1]

    valid_logits = torch.tensor(y_probs_val).cuda()
    valid_labels = torch.tensor(y_val).long().view(-1).cuda()
    test_logits = torch.tensor(y_probs_test).cuda()
    test_labels = torch.tensor(y_test).long().view(-1).cuda()
        
    return ((valid_logits, valid_labels), (test_logits, test_labels), n_class)
