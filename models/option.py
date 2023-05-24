import argparse

def opt():
    parser = argparse.ArgumentParser(description='SCTL')
    # Datasets
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Select the datasets')
    
    # Calibrator and Training Loss
    parser.add_argument('--cal', type=str, default='TS', 
                        help='TS : Temperature Scaling, ETS : Ensemble Temperature Scaling, CTS : Class-based Temeprature Scailing, PTS : Parameterized Temerature Scailng')
    parser.add_argument('--loss', type=str, default='CE', 
                        help='CE : Cross Entropy, LS : Label Smoothing loss, FL : Focal Loss, CL : scaling Class-wise Loss')
    
    # Hyper-parameter
    parser.add_argument('--n_iter', type=int, default=1000, 
                        help='Limit the max iter for optimizer')
    parser.add_argument('--lr', type=float, default=0.02, 
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.02, 
                        help='weight_decay')
    parser.add_argument('--norm', type=str, default='ND',
                        help='ND : Normal Distribution(standardization), CM : Centerized Min-max normalization ,MM : Min-Max Normalization')
    
    # Log
    parser.add_argument('--name', type=str, default='text.log',
                        help='Name of log')
    parser.add_argument('--trainlog', action='store_true',
                        help='Logging loss and measures during training')
    

    return parser.parse_args()
