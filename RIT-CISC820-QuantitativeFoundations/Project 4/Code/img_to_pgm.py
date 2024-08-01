import torchvision

test_img = torchvision.io.read_image("test_img.jpg")

test_img = torchvision.transforms.Grayscale()(test_img)
test_img = torchvision.transforms.Resize((112, 92))(test_img)

test_img = (test_img / 255).squeeze(0)

# print(test_img.dtype)
torchvision.utils.save_image(test_img, "test_img.png")

