rem "caffe_run2/caffe_best_model.exe" train --solver=proto/baseline/solver_ResNet34_m.prototxt --snapshot=proto/baseline/snapshot/ResNet34_m2_iter_400000.solverstate --gpu=all --log_name=ResNet34_m3 --log_dirs=proto/baseline/snapshot/log
"caffe_run2/caffe_best_model.exe" train --solver=proto/baseline/solver_ResNet50_m.prototxt --gpu=all --snapshot=proto/baseline/snapshot/ResNet50_m_iter_300000.solverstate --log_name=ResNet50_m --log_dirs=proto/baseline/snapshot/log