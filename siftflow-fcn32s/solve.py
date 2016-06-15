import caffe
import numpy as np

import score
import surgery

# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '../vgg16fc.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
# caffe.set_mode_gpu()
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
    score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
