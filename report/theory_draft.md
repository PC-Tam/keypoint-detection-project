# Theory Draft — Keypoint Detection: Phân tích bốn luồng tiếp cận

---

## 1. Tổng quan bài toán (Trọng Nguyên)

### 1.1 Keypoint Detection là gì?

**Keypoint Detection** (phát hiện điểm đặc trưng) là bài toán xác định các vị trí nổi bật trong ảnh — chẳng hạn góc cạnh, vùng blob, hay các cấu trúc cục bộ có tính phân biệt cao — sao cho các vị trí này có thể được định vị lại ổn định dưới các biến đổi hình học và quang học khác nhau (xoay, co giãn tỉ lệ, thay đổi ánh sáng).

Đây là bước nền tảng trong nhiều tác vụ thị giác máy tính: **matching ảnh** (tìm điểm tương ứng giữa hai khung nhìn), **nhận dạng đối tượng** (so khớp mẫu trong cơ sở dữ liệu), **tái tạo 3D** (Structure from Motion, stereo vision), và **tracking** đối tượng theo thời gian thực. Chất lượng của bước phát hiện keypoint ảnh hưởng trực tiếp đến độ chính xác và hiệu quả của toàn bộ pipeline xử lý phía sau.

### 1.2 Phân loại các phương pháp

Trong phạm vi dự án này, các phương pháp được tổ chức theo bốn luồng nghiên cứu chính:

| Luồng | Phương pháp tiêu biểu | Đặc điểm |
|---|---|---|
| **Hand-crafted** | SIFT (Phương pháp chính), Harris & LoG/DoH (Phương pháp phụ/Nền tảng)| Thiết kế thủ công dựa trên tri thức toán học, tính bất biến rõ ràng |
| **Real-time** | ORB (Phương pháp chính), FAST (Phương pháp phụ) | Tối ưu tốc độ, phù hợp hệ thống nhúng và ứng dụng thời gian thực |
| **ML Clustering** | BoVW (Phương pháp chính), K-Means/K-NN (Phương pháp phụ/bổ trợ)| Học không giám sát, xây dựng từ điển thị giác để biểu diễn toàn cục |
| **Deep Learning** | CNN, MobileNetV2, Grad-CAM | Tự động học đặc trưng end-to-end từ dữ liệu |

### 1.3 Mối quan hệ giữa bốn luồng trong pipeline thực tế

Bốn luồng trên không hoạt động độc lập mà có sự kết nối chặt chẽ trong một pipeline hoàn chỉnh:

- **Hand-crafted + Real-time** đảm nhận vai trò phát hiện và mô tả keypoint cục bộ (local descriptors), cung cấp đầu vào thô cho bước mã hoá thị giác.
- **BoVW (Bag of Visual Words)** nhận các descriptors đó, lượng tử hoá chúng bằng K-Means để xây dựng từ điển thị giác, từ đó tạo ra biểu diễn toàn cục (global representation) dạng histogram cho mỗi ảnh.
- **K-NN matching** hoạt động trên không gian biểu diễn toàn cục này để thực hiện truy vấn và so khớp ảnh.
- **CNN / MobileNetV2** cung cấp một hướng tiếp cận thay thế theo kiểu end-to-end, trong đó đặc trưng được học trực tiếp từ dữ liệu; **Grad-CAM** được dùng để trực quan hoá vùng ảnh mà mạng tập trung vào, đóng vai trò tương đương "keypoint saliency" ở mức giải thích mô hình.

### 1.4 Ứng dụng thực tế

Keypoint Detection là thành phần cốt lõi trong nhiều hệ thống thị giác máy tính hiện đại:

- **Image stitching (panorama):** ghép nhiều ảnh chụp liên tiếp thành ảnh toàn cảnh thông qua homography ước lượng từ các cặp keypoint tương ứng.
- **Object recognition:** nhận dạng và định vị đối tượng trong ảnh bằng cách so khớp đặc trưng cục bộ với mẫu lưu trữ.
- **Visual SLAM:** cho phép robot và phương tiện tự hành ước lượng vị trí và xây dựng bản đồ môi trường theo thời gian thực dựa trên tracking keypoint liên tục.
- **Image retrieval:** tìm kiếm ảnh tương tự trong cơ sở dữ liệu lớn, ứng dụng trong thương mại điện tử, bản quyền nội dung và địa lý hình ảnh.

---

## 2. Hand-crafted Features (Duy Quang)

Các phương pháp hand-crafted (đặc trưng thủ công) được xây dựng dựa trên các nguyên lý toán học minh bạch, không yêu cầu dữ liệu huấn luyện. Ưu điểm chính là tính giải thích được cao và khả năng hoạt động tốt trong môi trường có ít dữ liệu. Nhược điểm là hiệu năng bị giới hạn bởi các giả thiết thiết kế cứng (hard-coded priors).

### 2.1 Harris Corner Detector

#### Nguyên lý

Harris Corner Detector (Harris & Stephens, 1988) phát hiện góc dựa trên sự thay đổi cường độ ảnh theo mọi hướng. Ý tưởng cốt lõi: tại một điểm phẳng, cửa sổ trượt không thay đổi theo mọi hướng; tại cạnh, chỉ thay đổi theo một hướng; tại góc, thay đổi theo mọi hướng.

#### Ma trận cấu trúc (Second Moment Matrix)

Cho một cửa sổ $W$ đặt tại điểm $(x, y)$, ma trận cấu trúc $M$ được định nghĩa:

$$M = \sum_{(u,v) \in W} w(u,v) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

trong đó $I_x$, $I_y$ là đạo hàm ảnh theo chiều ngang và dọc, $w(u,v)$ là hàm cửa sổ (thường là Gaussian).

#### Hàm phản hồi Harris

Thay vì tính trực tiếp eigenvalue $\lambda_1, \lambda_2$ của $M$ (tốn kém), Harris đề xuất hàm phản hồi:

$$R = \det(M) - k \cdot (\text{trace}(M))^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

với $k \in [0.04, 0.06]$ là hằng số thực nghiệm. Phân loại:
- $R \gg 0$: góc (corner)
- $R \ll 0$: cạnh (edge)
- $|R| \approx 0$: vùng phẳng (flat)

#### Đặc điểm

Harris bất biến với phép xoay (rotation-invariant) vì $\det(M)$ và $\text{trace}(M)$ là bất biến xoay. Tuy nhiên, phương pháp này **không bất biến tỉ lệ** (not scale-invariant): kích thước cửa sổ cố định dẫn đến phản hồi khác nhau ở các mức thu phóng.

---

### 2.2 LoG và DoH — Phát hiện Blob

#### Laplacian of Gaussian (LoG)

LoG kết hợp làm mịn Gaussian để khử nhiễu và toán tử Laplacian để phát hiện vùng có biến đổi cường độ nhanh (blob). Hàm lõi:

$$\text{LoG}(x, y, \sigma) = -\frac{1}{\pi\sigma^4} \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Điểm keypoint được xác định tại các cực trị cục bộ trong không gian $(x, y, \sigma)$, trong đó $\sigma$ đóng vai trò tham số tỉ lệ. Điều này cho phép LoG phát hiện blob ở nhiều kích thước khác nhau, đạt tính **bất biến tỉ lệ**.

#### Difference of Gaussian (DoG) — xấp xỉ LoG

Vì LoG tính toán nặng, Lowe (2004) đề xuất xấp xỉ bằng Difference of Gaussian:

$$\text{DoG}(x, y, \sigma) = G(x, y, k\sigma) - G(x, y, \sigma)$$

DoG xấp xỉ $\sigma^2 \nabla^2 G$ và được sử dụng trong SIFT.

#### Determinant of Hessian (DoH)

DoH dựa trên ma trận Hessian của ảnh đã làm mịn:

$$H(x, y, \sigma) = \begin{bmatrix} L_{xx} & L_{xy} \\ L_{xy} & L_{yy} \end{bmatrix}$$

Keypoint tại các cực trị của $\det(H) = L_{xx}L_{yy} - L_{xy}^2$. DoH chính xác hơn LoG trong việc định vị blob và được dùng trong thuật toán SURF. Tuy nhiên, DoH nhạy cảm hơn với nhiễu so với LoG.

---

### 2.3 SIFT — Scale-Invariant Feature Transform

SIFT (Lowe, 2004) là một trong các thuật toán hand-crafted hoàn chỉnh và mạnh mẽ nhất, đạt tính bất biến với tỉ lệ, xoay, và một phần với biến đổi affine.

#### Bước 1 — Xây dựng Scale Space và phát hiện keypoint

Ảnh được tích chập với Gaussian tại nhiều mức $\sigma$ để tạo ra không gian tỉ lệ (scale space). Chuỗi DoG được tính giữa các mức liền kề. Keypoint là các cực trị cục bộ trong không gian 3D $(x, y, \sigma)$, tức là vượt trội so với 26 điểm lân cận (8 cùng tầng, 9 tầng trên, 9 tầng dưới).

#### Bước 2 — Lọc và tinh chỉnh keypoint

Các keypoint có phản hồi thấp (vùng phẳng) hoặc nằm trên cạnh (edge response cao, kiểm tra bằng tỉ số eigenvalue của Hessian) bị loại bỏ. Vị trí keypoint được tinh chỉnh đến độ chính xác sub-pixel bằng khai triển Taylor.

#### Bước 3 — Gán hướng (Orientation Assignment)

Histogram gradient hướng (36 bins, bước 10°) được tính trong vùng lân cận keypoint. Hướng chính là đỉnh histogram. Điều này đảm bảo tính bất biến xoay: descriptor sau đó được tính tương đối so với hướng này.

#### Bước 4 — Tính Descriptor

Vùng 16×16 pixel xung quanh keypoint được chia thành 4×4 sub-block. Mỗi sub-block tạo ra một histogram gradient 8 hướng → tổng $4 \times 4 \times 8 = 128$ chiều. Vector 128 chiều này là SIFT descriptor, sau đó được chuẩn hoá L2 để bất biến với biến đổi ánh sáng tuyến tính.

---

## 3. Real-time Features (Chí Tâm)

Các phương pháp real-time được thiết kế để giảm thiểu chi phí tính toán so với hand-crafted features truyền thống, nhằm đáp ứng yêu cầu xử lý theo thời gian thực trên phần cứng hạn chế (thiết bị nhúng, mobile, robot).

### 3.1 FAST — Features from Accelerated Segment Test

#### Nguyên lý

FAST (Rosten & Drummond, 2006) phát hiện góc bằng cách kiểm tra vòng tròn Bresenham bán kính 3 pixel (16 pixel xung quanh) tại điểm ứng viên $p$.

**Điều kiện góc:** tồn tại $n$ pixel liên tiếp trên vòng tròn đều sáng hơn $I_p + t$ hoặc đều tối hơn $I_p - t$, với $t$ là ngưỡng cường độ và $n$ thường là 12 (FAST-12).

#### Kiểm tra nhanh (High-speed test)

Để tăng tốc, trước tiên chỉ kiểm tra 4 pixel ở vị trí 1, 5, 9, 13. Nếu ít hơn 3 trong số này thoả điều kiện, điểm $p$ không thể là góc và bị bỏ qua ngay.

#### Non-maximum Suppression

Vì FAST có xu hướng phát hiện các keypoint liền kề nhau, bước NMS được áp dụng: giữ lại điểm có score cao nhất trong vùng lân cận, loại bỏ các điểm còn lại.

#### Hạn chế

FAST không tính toán descriptor, không bất biến tỉ lệ, và nhạy cảm với nhiễu ảnh. Những hạn chế này được giải quyết trong ORB.

---

### 3.2 ORB — Oriented FAST and Rotated BRIEF

ORB (Rublee et al., 2011) kết hợp FAST cho bước phát hiện và một biến thể của BRIEF cho bước mô tả, đồng thời bổ sung tính bất biến xoay và robustness với nhiễu.

#### Phát hiện keypoint — oFAST

ORB chạy FAST ở nhiều mức pyramid của ảnh để đạt xấp xỉ tính bất biến tỉ lệ. Tại mỗi mức, intensity centroid được dùng để gán hướng cho keypoint:

$$\theta = \text{atan2}(m_{01}, m_{10})$$

trong đó $m_{pq} = \sum_{x,y} x^p y^q I(x,y)$ là moment ảnh trong patch xung quanh keypoint.

#### Descriptor — rBRIEF (Rotated BRIEF)

BRIEF (Binary Robust Independent Elementary Features) xây dựng descriptor nhị phân bằng cách so sánh cường độ các cặp pixel ngẫu nhiên trong patch. ORB mở rộng BRIEF thành **rBRIEF**: các cặp điểm được xoay theo hướng $\theta$ của keypoint trước khi so sánh, đảm bảo bất biến xoay.

Descriptor nhị phân 256-bit cho phép tính khoảng cách Hamming thay vì Euclidean, giảm đáng kể thời gian matching.

#### Ưu điểm tổng hợp

- Không có bằng sáng chế (patent-free), phù hợp ứng dụng thương mại.
- Tốc độ vượt trội so với SIFT và SURF trong hầu hết môi trường thực tế.
- Độ chính xác matching cạnh tranh được với SURF trong điều kiện thay đổi góc nhìn và ánh sáng vừa phải.

---

## 4. ML Clustering — Bag of Visual Words (Trọng Nguyên)

Bag of Visual Words (BoVW) là framework học máy không giám sát, lấy cảm hứng từ mô hình Bag of Words trong xử lý ngôn ngữ tự nhiên. Thay vì làm việc trực tiếp với local descriptors (chiều cao, số lượng thay đổi theo ảnh), BoVW chuyển đổi mỗi ảnh thành một vector histogram có kích thước cố định, tạo điều kiện cho các thuật toán phân loại và truy vấn chuẩn.

### 4.1 K-Means Clustering

#### Mục tiêu

K-Means được dùng để xây dựng **từ điển thị giác** (visual vocabulary) từ tập hợp lớn local descriptors trích xuất từ corpus ảnh huấn luyện.

#### Thuật toán

Cho tập $N$ descriptors $\{x_i\} \in \mathbb{R}^d$ và số cụm $K$, K-Means tìm tâm cụm $\{\mu_k\}$ tối thiểu hoá hàm mục tiêu:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Vòng lặp EM:**
1. **E-step (Assignment):** gán mỗi điểm vào cụm có tâm gần nhất: $c_i = \arg\min_k \|x_i - \mu_k\|^2$
2. **M-step (Update):** cập nhật tâm cụm: $\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$

Lặp đến khi hội tụ (tâm cụm không thay đổi đáng kể).

#### Chọn K

$K$ là siêu tham số quan trọng: $K$ quá nhỏ dẫn đến từ điển không đủ biểu đạt (underfitting); $K$ quá lớn tăng chi phí tính toán và có thể gây overfitting. Phương pháp Elbow và Davies-Bouldin Index thường được dùng để lựa chọn $K$ phù hợp.

---

### 4.2 K-Nearest Neighbors (K-NN)

#### Vai trò trong pipeline BoVW

Sau khi mỗi ảnh được biểu diễn bằng histogram BoVW, K-NN thực hiện **image retrieval**: tìm $K$ ảnh trong cơ sở dữ liệu có vector biểu diễn gần nhất với ảnh truy vấn.

#### Độ đo khoảng cách

Với histogram BoVW, các độ đo phổ biến gồm:

- **Cosine similarity:** $\text{sim}(q, d) = \frac{q \cdot d}{\|q\|\|d\|}$ — phù hợp với vector thưa (sparse).
- **Chi-squared distance:** $\chi^2(q, d) = \sum_k \frac{(q_k - d_k)^2}{q_k + d_k}$ — nhạy cảm hơn với sự khác biệt tại các từ thị giác có tần suất thấp.

#### Đặc điểm

K-NN là thuật toán **lazy learning** (không có pha training), đơn giản và hiệu quả. Nhược điểm chính là chi phí tìm kiếm tuyến tính $O(N)$ theo kích thước cơ sở dữ liệu; có thể tăng tốc bằng KD-Tree, Ball Tree, hoặc approximate search (FAISS, Annoy).

---

### 4.3 BoVW Pipeline tổng thể

BoVW pipeline bao gồm hai giai đoạn chính:

#### Giai đoạn xây dựng (Offline)

```
Corpus ảnh
    → Trích xuất local descriptors (SIFT / ORB / ...)
    → K-Means clustering trên toàn bộ descriptors
    → Visual vocabulary V = {μ₁, μ₂, ..., μ_K}
    → Mỗi ảnh trong cơ sở dữ liệu:
        → Gán mỗi descriptor vào visual word gần nhất (hard assignment)
        → Xây dựng histogram tần suất K chiều
        → (Tuỳ chọn) TF-IDF weighting
    → Lưu trữ histogram database
```

#### Giai đoạn truy vấn (Online)

```
Ảnh truy vấn
    → Trích xuất descriptors
    → Ánh xạ vào vocabulary V
    → Tạo histogram truy vấn q
    → K-NN tìm kiếm trong histogram database
    → Trả về K ảnh tương tự nhất
```

#### TF-IDF Weighting

Tương tự NLP, visual word phổ biến trên nhiều ảnh mang ít thông tin phân biệt hơn. TF-IDF (Term Frequency–Inverse Document Frequency) điều chỉnh trọng số:

$$w_{ik} = \text{tf}_{ik} \cdot \log\frac{N}{n_k}$$

trong đó $\text{tf}_{ik}$ là tần suất visual word $k$ trong ảnh $i$, $N$ là tổng số ảnh, $n_k$ là số ảnh chứa visual word $k$.

---

## 5. Deep Learning (Lan Thanh)

Khác với các phương pháp truyền thống dựa trên đặc trưng thiết kế thủ công, deep learning cho phép mạng nơ-ron **tự động học** các đặc trưng phân cấp trực tiếp từ dữ liệu. Điều này đặc biệt có lợi khi không gian biến đổi của ảnh phức tạp và khó mô hình hoá bằng công thức toán học tường minh.

### 5.1 CNN Feature Extraction

#### Kiến trúc tổng quát

Convolutional Neural Network (CNN) xây dựng biểu diễn đặc trưng phân cấp thông qua các khối chức năng:

- **Convolutional layer:** tích chập với bộ lọc học được $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$, phát hiện cấu trúc cục bộ.
- **Activation (ReLU):** $f(x) = \max(0, x)$, đưa vào tính phi tuyến.
- **Pooling layer:** giảm chiều không gian (spatial downsampling), tăng receptive field và tính bất biến dịch chuyển cục bộ.
- **Batch Normalization:** ổn định phân phối activation qua các layer, tăng tốc huấn luyện.

#### Đặc trưng phân cấp

Các lớp đầu của CNN học đặc trưng cấp thấp (cạnh, gradient hướng), tương tự Harris hay SIFT. Các lớp sâu hơn kết hợp thành đặc trưng ngữ nghĩa cấp cao (texture, bộ phận đối tượng, khái niệm). Feature map tại các lớp trung gian có thể được dùng như descriptor tổng quát cho downstream tasks (transfer learning).

#### Transfer Learning

CNN được pretrain trên tập dữ liệu lớn (ví dụ ImageNet) học được biểu diễn đặc trưng phong phú. Feature vector trích từ các lớp fully-connected hoặc global average pooling của mạng pretrain thường vượt trội so với hand-crafted descriptors trên nhiều tác vụ recognition và retrieval.

---

### 5.2 MobileNetV2

#### Động lực thiết kế

MobileNetV2 (Sandler et al., 2018) được thiết kế cho ứng dụng di động và nhúng, tập trung tối ưu cân bằng giữa độ chính xác và chi phí tính toán (FLOPs, số tham số).

#### Depthwise Separable Convolution

Thay vì convolution tiêu chuẩn, MobileNet sử dụng tích chập tách tầng:

1. **Depthwise convolution:** mỗi kênh đầu vào được tích chập độc lập bằng một bộ lọc $k \times k$ riêng.
2. **Pointwise convolution (1×1):** tổ hợp tuyến tính các kênh để tạo đặc trưng mới.

Phương pháp này giảm đáng kể số phép nhân-cộng so với convolution tiêu chuẩn trong khi duy trì khả năng biểu diễn.

#### Inverted Residuals và Linear Bottleneck

Nét đặc trưng của MobileNetV2 là **inverted residual block**: khác với ResNet mở rộng chiều rồi thu hẹp, MobileNetV2 thu hẹp (bottleneck) trước, thực hiện depthwise conv, rồi mở rộng trở lại. Skip connection được áp dụng trực tiếp trên bottleneck (chiều thấp), tiết kiệm bộ nhớ. **Linear bottleneck** (không dùng ReLU ở lớp output của block) giúp bảo toàn thông tin không gian đặc trưng tại chiều thấp.

#### Ứng dụng trong dự án

MobileNetV2 được dùng như **feature extractor**: thay thế classifier head (lớp cuối) bằng global average pooling để thu được vector đặc trưng compact, dùng cho image retrieval và so sánh với pipeline BoVW.

---

### 5.3 Grad-CAM — Gradient-weighted Class Activation Mapping

#### Mục tiêu

Grad-CAM (Selvaraju et al., 2017) là phương pháp **giải thích mô hình** (explainability), trực quan hoá vùng ảnh mà CNN tập trung vào khi đưa ra dự đoán cho một lớp cụ thể. Trong ngữ cảnh keypoint detection, Grad-CAM cung cấp "bản đồ nổi bật" (saliency map) tương đương, chỉ ra vùng thị giác có tính phân biệt cao theo học của mạng.

#### Cơ chế tính toán

Cho lớp mục tiêu $c$ và feature map $A^k$ của lớp convolution cuối cùng (kích thước $H \times W$):

**Bước 1 — Tính trọng số tầm quan trọng:**
$$\alpha_k^c = \frac{1}{HW} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

trong đó $y^c$ là logit của lớp $c$ trước softmax, $\frac{\partial y^c}{\partial A_{ij}^k}$ là gradient ngược qua backpropagation.

**Bước 2 — Tính heatmap:**
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

ReLU loại bỏ các vùng ảnh hưởng tiêu cực (làm giảm score lớp $c$), chỉ giữ vùng hỗ trợ quyết định.

**Bước 3 — Upsampling:** heatmap được nội suy về kích thước ảnh gốc để trực quan hoá.

#### Ý nghĩa trong pipeline

Grad-CAM không thay thế keypoint detector truyền thống, mà cung cấp góc nhìn **interpretability**: so sánh xem mạng CNN tập trung vào vùng nào so với các keypoint do SIFT, ORB phát hiện, từ đó đánh giá sự tương đồng và khác biệt giữa hai paradigm.

---

## 6. So sánh bốn luồng tiếp cận

### 6.1 Bảng so sánh tổng hợp

| Tiêu chí | Hand-crafted (Harris, SIFT) | Real-time (FAST, ORB) | ML Clustering (BoVW) | Deep Learning (CNN/MobileNetV2) |
|---|---|---|---|---|
| **Yêu cầu dữ liệu** | Không cần | Không cần | Cần corpus không nhãn | Cần dữ liệu có nhãn (hoặc pretrain) |
| **Tính bất biến** | Tỉ lệ, xoay (SIFT) | Xoay (ORB), xấp xỉ tỉ lệ | Phụ thuộc descriptor đầu vào | Học được từ dữ liệu, tổng quát hơn |
| **Tốc độ** | Chậm–vừa | Rất nhanh | Trung bình (offline heavy) | Phụ thuộc phần cứng (GPU) |
| **Khả năng giải thích** | Cao (công thức tường minh) | Cao | Trung bình (clustering) | Thấp (Grad-CAM hỗ trợ giải thích) |
| **Biểu diễn đặc trưng** | Cục bộ (local descriptor) | Cục bộ (binary) | Toàn cục (global histogram) | Toàn cục (embedding vector) |
| **Độ chính xác retrieval** | Tốt với ảnh đơn giản | Thấp hơn SIFT | Tốt với $K$ và vocabulary phù hợp | Tốt nhất khi có dữ liệu đủ lớn |
| **Chi phí triển khai** | Thấp | Rất thấp | Trung bình | Cao (GPU, inference latency) |

### 6.2 Phân tích theo chiều sâu

#### Về tính bất biến

SIFT đạt tính bất biến tỉ lệ và xoay nhờ thiết kế có chủ ý qua scale space và orientation assignment. ORB xấp xỉ bất biến tỉ lệ qua image pyramid nhưng kém SIFT trong điều kiện scale change lớn. CNN học tính bất biến từ dữ liệu, có thể tổng quát hơn nhưng phụ thuộc mạnh vào sự đa dạng của tập huấn luyện.

#### Về biểu diễn toàn cục vs cục bộ

Hand-crafted và real-time methods tạo ra **local descriptors** — phù hợp cho geometric matching nhưng không thể so sánh trực tiếp hai ảnh toàn cục. BoVW và CNN embedding tạo ra **global representation** — phù hợp cho image retrieval theo nghĩa tìm kiếm ngữ nghĩa toàn ảnh.

#### Về khả năng kết hợp

Pipeline BoVW là cầu nối tự nhiên giữa hai thế giới: tận dụng sức mạnh của local descriptors (hand-crafted hoặc real-time) rồi chuyển đổi thành global representation phục vụ matching. CNN embedding có thể coi là bước tiến hoá tiếp theo: loại bỏ cả bước thiết kế descriptor lẫn bước clustering thủ công, thay bằng một mạng end-to-end.

#### Lựa chọn thực tế

- **Tài nguyên hạn chế, không có GPU:** ORB + BoVW là lựa chọn cân bằng tốt.
- **Yêu cầu độ chính xác cao, có GPU:** CNN embedding (MobileNetV2) với vector search (FAISS).
- **Cần giải thích được (explainability):** Hand-crafted methods hoặc Grad-CAM trên CNN.
- **Real-time robotics:** FAST / ORB với tracking trực tiếp trên local descriptors.

### 6.3 Vị trí của bốn luồng trong landscape nghiên cứu

Bốn luồng tiếp cận không loại trừ nhau mà phản ánh sự tiến hoá của lĩnh vực: từ thiết kế dựa trên tri thức chuyên gia (hand-crafted) → tối ưu hoá tốc độ thực thi (real-time) → học biểu diễn không giám sát (BoVW) → học biểu diễn end-to-end có giám sát (deep learning). Trong thực tế, các hệ thống tiên tiến thường kết hợp nhiều luồng: ví dụ dùng CNN để học descriptor thay thế SIFT trong pipeline BoVW (NetVLAD), hoặc dùng ORB để khởi tạo tracking trước khi tinh chỉnh bằng deep features.

---

## Tài liệu tham khảo

### Phương pháp Hand-crafted

[1] C. Harris và M. Stephens. "A Combined Corner and Edge Detector." *Proceedings of the Alvey Vision Conference*, 1988.

[2] T. Lindeberg. "Feature Detection with Automatic Scale Selection." *International Journal of Computer Vision*, 1998.

[3] D. G. Lowe. "Object Recognition from Local Scale-Invariant Features." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1999.

[4] D. G. Lowe. "Distinctive Image Features from Scale-Invariant Keypoints." *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, 2004.

[5] H. Bay, T. Tuytelaars và L. Van Gool. "SURF: Speeded Up Robust Features." *European Conference on Computer Vision (ECCV)*, 2006.

### Phương pháp Real-time

[6] E. Rosten và T. Drummond. "Machine Learning for High-speed Corner Detection." *European Conference on Computer Vision (ECCV)*, 2006. DOI: [10.1007/11744023_34](http://doi.org/10.1007/11744023_34)

[7] M. Calonder, V. Lepetit, C. Strecha và P. Fua. "BRIEF: Binary Robust Independent Elementary Features." *ECCV*, 2010. DOI: [10.1007/978-3-642-15561-1_56](http://doi.org/10.1007/978-3-642-15561-1_56)

[8] E. Rublee, V. Rabaud, K. Konolige và G. Bradski. "ORB: An Efficient Alternative to SIFT or SURF." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2011. DOI: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544)

### ML Clustering — Bag of Visual Words

[9] J. MacQueen. "Some Methods for Classification and Analysis of Multivariate Observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281–297, 1967.

[10] T. Cover và P. Hart. "Nearest Neighbor Pattern Classification." *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967. DOI: [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)

### Deep Learning

[11] M. Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018. DOI: [10.1109/CVPR.2018.00474](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

[12] R. R. Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017. DOI: [10.1109/ICCV.2017.74](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)

### Sách tham khảo

[13] G. Bradski và A. Kaehler. *Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library*. O'Reilly Media, 2016. Chương 16: Feature Detection and Matching.

[14] R. C. Gonzalez và R. E. Woods. *Digital Image Processing*. 4th ed. Pearson, 2018. Chương 11: Representation and Description; Chương 12: Object Recognition.

---
