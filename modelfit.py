'''
fit a model based on the param
'''
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import RMSprop

def fit_model(model, train_generator, validation_generator, epochs, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(decay=1e-4, lr=0.01), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0, mode='auto', restore_best_weights=True)
    callbacks_list = [early_stop]

    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=len(train_generator.filenames) // batch_size,
                                        epochs=epochs,
                                        validation_data=validation_generator,
                                        validation_steps=len(train_generator.filenames) // batch_size,
                                        callbacks=callbacks_list)

    return model, model_history