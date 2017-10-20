import chainer
from chainer import training, iterators, serializers
from chainer.training import extensions
# import sys
# sys.path.append("../dl_utils")
# from dl_utils.frameworks.chainer.snapshot_best import snapshot_best
# from iterator import MyIterator
# from extentions import TestLossReport

default_params = {
    "gpus": [],
    "test_run": False,
    "path_results": "/tmp/chainer",
    "batch_size": 32
}


def train(model, train, test=None, params={}):
    params_local = default_params
    params_local.update(params)
    gpus = params_local["gpus"]

    if len(gpus) > 1:
        print("multiple gpus not implemented yet")
        exit(-1)

    if len(gpus) == 1:
        chainer.cuda.get_device(gpus[0]).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = iterators.SerialIterator(train, batch_size=params_local["batch_size"], shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=params_local["batch_size"], repeat=False, shuffle=False)

    if len(gpus) == 1:
        updater = training.StandardUpdater(train_iter, optimizer, device=gpus[0])
    else:
        updater = training.StandardUpdater(train_iter, optimizer)

    #if params["test_run"] is True:
        #trainer = training.Trainer(updater, stop_trigger=(1, 'epoch'), out=params["path_results"])
    #else:
        #trainer = training.Trainer(updater, stop_trigger=stop_trigger, out=params["path_results"])
    trainer = training.Trainer(updater, stop_trigger=(20, 'epoch'), out=params_local["path_results"])
    #if len(gpus) == 1:
        #trainer.extend(extensions.Evaluator(test_iter, model, device=gpus[0], eval_func=model.test_eval_func))
    #else:
        #trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.test_eval_func))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename='snapshot_last.npz'), trigger=(1, 'epoch'))
    # trainer.extend(snapshot_best(filename='snapshot_best.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.observe_value("alpha", lambda _: optimizer.alpha))
    trainer.extend(extensions.ExponentialShift("alpha", 0.99), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'alpha', 'elapsed_time']))

    trainer.run()
