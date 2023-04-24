from argparse import Namespace
parser1 = Namespace()
parser1.weights = 'runs/train/bowen-run-27-new/weights/best.pt'
# parser1.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
# opt1 = parser1.parse_args()
print(parser1.weights)