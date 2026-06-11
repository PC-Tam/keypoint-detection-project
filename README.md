# Phân loại Bệnh Glaucoma bằng Computer Vision Truyền thống & Machine Learning

Dự án này triển khai một pipeline để phân loại hình ảnh võng mạc (fundus) thành Bình thường (Normal) hoặc Mắc bệnh Glaucoma bằng các kỹ thuật Computer Vision truyền thống (Harris Corner, FAST, ORB) và các mô hình Machine Learning (SVM, Random Forest, KNN, Naive Bayes), mà không cần sử dụng Deep Learning.

## Các tính năng
- **Trích xuất đặc trưng truyền thống:** Sử dụng Harris Corner Detection, FAST và ORB để xác định các điểm đặc trưng (keypoints).
- **Bag of Visual Words (BoVW):** Chuyển đổi các descriptor cục bộ của ORB thành các đặc trưng hình ảnh tổng thể thông qua phân cụm K-Means.
- **Các mô hình Machine Learning:** Huấn luyện các mô hình SVM, Random Forest, KNN và Gaussian Naive Bayes dựa trên các đặc trưng đã trích xuất.
- **Giao diện người dùng (UI) tương tác:** Ứng dụng Streamlit với thiết kế UI/UX hiện đại dùng để chẩn đoán và trực quan hóa dữ liệu từ đầu đến cuối.

## Cài đặt & Thiết lập

1. Đảm bảo bạn đã cài đặt Python.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Tập dữ liệu (Dataset)

Dự án này được xây dựng để hoạt động với tập dữ liệu **ACRIMA**, hoặc tùy chọn là **RIM-ONE DL**.

### Tải tự động
Nếu bạn đã thiết lập token Kaggle API (`~/.kaggle/kaggle.json`), pipeline sẽ tự động tải tập dữ liệu ACRIMA khi bạn chạy lệnh `run_pipeline.py`.

### Tải thủ công
Nếu tải tự động thất bại, vui lòng làm theo các bước sau:
1. Tải tập dữ liệu ACRIMA từ [Kaggle](https://www.kaggle.com/datasets/mloey1/acrima).
2. Giải nén tập dữ liệu.
3. Đặt tất cả các tệp hình ảnh võng mạc `.jpg` / `.png` trực tiếp vào thư mục `data/raw/ACRIMA/images/`.

*Lưu ý về nhãn (labels):* Theo mặc định, tập dữ liệu ACRIMA đặt tên các hình ảnh bị Glaucoma có chứa chuỗi `_g_` trong tên tệp, còn hình ảnh bình thường thì không có `_g_`. Pipeline sử dụng quy ước đặt tên này để tự động gán nhãn cho ảnh.

## Chạy Pipeline

Để chạy toàn bộ quá trình (bao gồm kiểm tra tập dữ liệu, tiền xử lý, trích xuất đặc trưng, huấn luyện mô hình ML, đánh giá và lưu kết quả):

```bash
python src/run_pipeline.py
```

Pipeline sẽ thực hiện các bước sau:
- Kiểm tra tập dữ liệu và trích xuất thông tin.
- Tiền xử lý hình ảnh (Trích xuất kênh màu Xanh lá, áp dụng bộ lọc cân bằng biểu đồ CLAHE).
- Trích xuất các đặc trưng Harris, FAST và ORB (kết hợp với BoVW).
- Huấn luyện các mô hình SVM, RF, KNN và GNB trên tất cả các loại đặc trưng.
- Đánh giá các mô hình và lưu lại các chỉ số, báo cáo, và ma trận nhầm lẫn (confusion matrices) vào thư mục `outputs/`.

## Phân tích Kết quả

Sau khi pipeline chạy xong, hãy kiểm tra thư mục `outputs/`:
- **`outputs/reports/results.csv`**: Chứa các chỉ số Accuracy, Precision, Recall, F1-Score, và Specificity cho từng sự kết hợp giữa đặc trưng và mô hình.
- **`outputs/reports/classification_report.txt`**: Chi tiết báo cáo phân loại.
- **`outputs/figures/`**: Chứa các biểu đồ ma trận nhầm lẫn và các hình ảnh minh họa keypoint.

*Đánh giá hiệu suất:* Đối với các phương pháp Computer Vision truyền thống (không dùng Deep Learning), độ chính xác (Accuracy) hoặc F1-Score trên 70% được coi là mức chấp nhận được, và trên 80% là mức tốt. Deep Learning thường mang lại hiệu suất cao hơn, nhưng các mô hình truyền thống này sẽ giúp ta hiểu rõ hơn về các đặc trưng hình ảnh cụ thể của bệnh Glaucoma.

## Giải thích về Đặc trưng
- **Harris Corners:** Thuật toán phát hiện các góc trong hình ảnh. Chúng tôi tổng hợp số lượng các góc, số liệu thống kê phản hồi (trung bình, lớn nhất, độ lệch chuẩn) và sự phân bố không gian của chúng.
- **FAST Keypoints:** Thuật toán phát hiện góc tương tự Harris nhưng với tốc độ nhanh hơn. Chúng tôi xây dựng một vector có kích thước cố định để tóm tắt các điểm đặc trưng này.
- **ORB + BoVW:** ORB trích xuất các điểm đặc trưng và descriptor bất biến với phép quay và tỷ lệ phóng to/thu nhỏ. Phương pháp BoVW (Bag of Visual Words) sẽ phân nhóm các descriptor này thành "các từ vựng trực quan" để tạo ra một biểu đồ histogram đại diện cho toàn bộ hình ảnh, mang lại những mô tả về kết cấu (texture) phong phú và đa dạng hơn so với các phương pháp thống kê góc đơn thuần.

## Chạy giao diện Demo (UI)

Bạn có thể chạy ứng dụng web tương tác để tải ảnh lên và xem trực tiếp các đặc trưng cũng như kết quả chẩn đoán theo thời gian thực (với giao diện đã được nâng cấp thiết kế chuyên nghiệp):

```bash
python -m streamlit run app.py
```
