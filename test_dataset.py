
from PIL import Image
import os

os.makedirs("test_images", exist_ok=True)
for i in range(2):
    img_path = f"test_images/img{i}.jpg"
    if not os.path.exists(img_path):  # Only create if not already present
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        img.save(img_path)

from dataset import get_loader
from vocab import DummyVocab
from torchvision import transforms

vocab = DummyVocab()
image_paths = [f"test_images/img{i}.jpg" for i in range(2)]
captions = [['a', 'man', 'riding', 'bike'], ['a', 'man', 'on', 'street']]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

loader = get_loader(image_paths, captions, vocab, transform, batch_size=2)

for images, caps in loader:
    print("✅ Images shape:", images.shape)
    print("✅ Captions shape:", caps.shape)
    print("🔢 First caption indices:", caps[0])
    break
