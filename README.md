
# Wood Anomaly Detection

Bu proje tahtalar üzerindeki anomalileri tespit eden önceden eğitilmiş 3 farklı yapay zeka modelinin frontend kullanarak gerçeklenmesini içerir.

## ÖNEMLİ NOT
Proje içindeki .pth dosyaları çok büyük olduğundan bazıları doğru olarak yüklenmemiş olabilir. Vereceğim drive linki üzerinden model ağırlıklarını klasörlere koyduktan sonra rahatlıkla çalıştırabilirsiniz.

### Kullanılan Modeller
- **GLASS** 
- **EfficientAD**
- **INP-Former**

## Gereksinimler

- Node.js (React için)
- Python 3.x
- `pip` paket yöneticisi


## Teknolojiler

- **Frontend**: React.js
- **Backend**: Python (Flask)


## Kurulum

### Frontend
```bash
npm install
npm start
```
Bu adımlardan sonra http://localhost:3000/ adresinde frontend ayağa kalkacaktır.

### Backend

```bash
cd .\backend\
pip install -r requirements.txt
python app.py
```
Bu adımlardan sonra http://localhost:5000 adresinde backend ayağa kalkacaktır.
