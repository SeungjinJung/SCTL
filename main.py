import logging
from models import calibrator, calibrator_mapping, dataloader, dataset_mapping, loss_mapping, opt

def main():
    args = opt()
    
    # We use abbreviations corresponding to data paths. 
    # More details of dataset abbreviations refer to the constant 'DATASET' in SCTL/utils (Line 9).
    datasets = dataset_mapping(args.dataset)
    
    # Call the logger for recording observations.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('log/'+args.name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    args.logger = logger

    for data in datasets:
        args.data = data
        # The dataset consists of output logits from a classifier pretrained on the training dataset.
        # The dataset is separated into the validation dataset and the test dataset.
        # The validation dataset is used to train the calibrator, and the test dataset is used for evaluation.
        (valid_logits, valid_labels), (test_logits, test_labels) = dataloader(args)

        model = calibrator(args)
        model.cuda()

        # Train a calibrator on valid dataset, such as TS, ETS, PTS, and CTS.
        model.train(valid_logits, valid_labels)

        # Evaluate a calibrator on test dataset.
        model.evaluate(test_logits,test_labels)

if __name__ == '__main__':
    main()
