import torch
print(torch.cuda.is_available())         # GPU kullanılabilir mi? True olmalı
print(torch.cuda.get_device_name(0))     # GPU'nun marka/modeli (örneğin RTX 4050)
