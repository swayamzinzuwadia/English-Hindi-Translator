import matplotlib.pyplot as plt

def evaluationPlotAccuracy(model):
    plt.plot(model.history["accuracy"], color="blue", label="accuracy")
    plt.plot(model.history["val_accuracy"], color="red", label="val_accuracy")
    plt.title("Training And Validation Accuracy Score")
    plt.xlabel("accuracy")
    plt.ylabel("epochs")
    plt.grid()
    plt.legend()
    plt.show()

def evaluationPlotLoss(model):
    plt.plot(model.history["loss"], color="blue", label="loss")
    plt.plot(model.history["val_loss"], color="red", label="val_loss")
    plt.title("Training And Validation Loss Score")
    plt.xlabel("loss")
    plt.ylabel("epochs")
    plt.grid()
    plt.legend()
    plt.show()
