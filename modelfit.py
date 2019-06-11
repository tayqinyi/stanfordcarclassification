'''
fit a model based on the param
'''
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import RMSprop

def fit_model(model, train_generator, validation_generator, epochs, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    callbacks_list = [early_stop, reduce_lr]

    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=len(train_generator.filenames) // batch_size,
                                        epochs=epochs,
                                        validation_data=validation_generator,
                                        validation_steps=len(train_generator.filenames) // batch_size,
                                        callbacks=callbacks_list)

    return model, model_history