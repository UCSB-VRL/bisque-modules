from src.vid_inference import *

def run_module(vid, k, model_weight, batch_size, num_workers, outfile):
    predict(vid, k, model_weight, batch_size, num_workers, outfile)

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int)
    p.add_argument('--vid', type=str)
    p.add_argument('--k', type=int, help='run inference on every "kth" frame')
    p.add_argument('--num_workers', type=int)
    p.add_argument('--model_weight', type=str)
    p.add_argument('--outfile', type=str)
    args = p.parse_args()

    batch_size = args.batch_size
    vid = args.vid
    k = args.k
    num_workers = args.num_workers
    model_weight = args.model_weight
    outfile = args.outfile

    run_module(vid, k, model_weight, batch_size, num_workers, outfile)