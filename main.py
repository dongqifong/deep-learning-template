if __name__ == "__main__":
    from trainer import Trainer
    from model import build_model
    data_source = "data/"
    preprocessor = None
    model_params = {"x_size":5,"n_channels":10,"dropout_p":0.1}
    trainer_parameters = {"batch_size":20, "epochs":30}

    model = build_model(**model_params)
    trainer = Trainer(data_source=data_source, preprocessor=preprocessor, model=model, **trainer_parameters)
    trainer.train()
    print(trainer.export_log_model())

    import numpy as np
    from predictor import Predictor

    m = {"model":trainer.model}
    predictor = Predictor(**m)
    x = np.random.random((7,5))
    y = predictor.predict(x)
    print(y)