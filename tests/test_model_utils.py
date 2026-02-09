from PIL import Image

from src.utils.model_utils import SimpleCNN, predict_image


def test_predict_image_output():
    model = SimpleCNN(num_classes=2)
    classes = ["Cat", "Dog"]
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    result = predict_image(model, img, classes)
    assert "label" in result
    assert "probabilities" in result
    assert set(result["probabilities"].keys()) == set(classes)
