# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về cả bức ảnh. Vì rank=1 nghĩa là cab (taxi) đứng đầu trong các lớp mà mô hình dự đoán các xe taxi cho ảnh traffic. Record không có tọa độ để xác định vị trí của một chiếc xe cụ thể.
3. Danh sách lớp đến từ taxonomy ImageNet-1K đi kèm checkpoint yolo11n-cls.pt
4. Ba trường giúp tra cứu nhãn chính xác. Cùng một mã số có thể chỉ các lớp khác nhau trong hai taxonomy.
5. Guideline cần quy định cách chọn chủ thể chính và xử lý trường hợp khó phân xử.
6. Ground truth cần được người gán nhãn xác định bằng cách xem ảnh và áp dụng guideline, vì dự đoán xếp hạng cao nhất vẫn có thể sai.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Tính từ góc góc trên trái của ảnh, góc trên trái của hộp nằm 385.33 pixel sang phải và 69.24 pixel xuống dưới. Góc dưới phải nằm 498.92 pixel sang phải và 348.92 pixel xuống dưới. Theo JSON, hộp rộng 113.58 pixel và cao 279.68 pixel.
3. Ở ngưỡng 0.35 có 11 prediction cho ảnh kitchen. Lọc 11 record theo score ≥ 0.60 thì còn 6 prediction.Đối chiếu số 6 với dòng threshold=0.60 trong notebook để trích đúng kết quả của ô so sánh.
4. Reviewer cần đối chiếu các hộp mới với ảnh để xem mỗi hộp có bao một vật thể thật hay không, lớp dự đoán có đúng không và hộp có ôm sát vật thể không. Nếu nhiều hộp cùng chỉ một vật thể, reviewer cần kiểm tra xem có dự đoán trùng hay không.
5. Vẽ hình chữ nhật nhỏ nhất bao trọn phần nhìn thấy của một vật thể. Đặt bốn cạnh sát phần ngoài cùng của vật thể và giữ hộp trong khung ảnh.
6. Ở mép trái ảnh, ngay cạnh bàn, mô hình vẽ một hộp person 0.61 quanh vùng chỉ thấy một phần cánh tay và bàn tay. Nếu guideline chưa nói rõ mức độ nhìn thấy tối thiểu để gán nhãn person, tôi sẽ hỏi Lab Coach xem có cần gán nhãn cho trường hợp chỉ thấy tay hay không. Khi có câu trả lời, có thể bổ sung quy tắc chung để mọi annotator xử lý nhất quán các ảnh tương tự.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. Polygon cho biết đường viền của vật thể. Box là hình chữ nhật bao quanh người nên chứa cả phần nền, chẳng hạn khoảng trống giữa hai chân. Polygon bám theo đường viền đầu, vai, tay và từng chân, giúp xác định vùng ảnh thuộc về người chính xác hơn.
3. instance_id dùng để phân biệt từng đối tượng trong kết quả phân đoạn. Ví dụ, kitchen-001 và kitchen-009 đều có lớp person, nhưng là hai đối tượng được mô hình dự đoán riêng. class_id xác định loại đối tượng, còn tracking ID dùng để theo dõi cùng một đối tượng qua nhiều khung hình. instance_id ở đây chỉ nhận diện đối tượng trong output của bài lab.
4. Quy tắc vẽ biên mask đề xuất. Đặt các điểm polygon dọc theo đường viền nhìn thấy của vật thể, thêm điểm ở chỗ biên đổi hướng và loại phần nền hoặc vật thể khác khỏi mask.
5. Trường hợp khó trên ảnh kitchen. Chiếc bát ở góc dưới bên trái che một phần mặt bàn. Tôi sẽ áp dụng guideline nếu đã quy định mask chỉ bao phần nhìn thấy. Khi đó, tôi tách vùng chiếc bát khỏi mask của bàn và dừng biên mask bàn tại chỗ tiếp giáp với bát. Nếu guideline chưa quy định cách xử lý phần bị che, tôi sẽ hỏi Lab Coach để cả nhóm gán nhãn nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho cả ảnh. | Trong classification_top5.png, ảnh traffic có nhiều xe buýt và ô tô, còn JSON xếp cab hạng 1. Việc chọn một lớp cho cả cảnh cần tiêu chí rõ ràng. | Xem toàn ảnh và chọn lớp theo tiêu chí xác định chủ thể chính trong guideline. | Kiểm tra lớp đã chọn có phù hợp với ảnh và tiêu chí trong guideline không. |
| Phát hiện vật thể | Một nhãn lớp và một hộp chữ nhật cho mỗi vật thể cần gán nhãn. | Trong detection_predictions.png, hộp person 0.61 ở mép trái ảnh kitchen bao vùng chỉ thấy một phần cánh tay và bàn tay. | Áp dụng quy tắc cho vật thể bị cắt mép. Nếu chưa rõ phần nhìn thấy có đủ để gán nhãn, hỏi Lab Coach. | Kiểm tra quyết định gán nhãn. Nếu có box, kiểm tra box có ôm sát phần nhìn thấy theo guideline không. |
| Instance segmentation | Một nhãn lớp và một mask riêng cho mỗi vật thể cần gán nhãn. | Trong segmentation_prediction.png, chiếc bát ở góc dưới bên trái ảnh kitchen che một phần mặt bàn, nên cần xác định biên giữa bát và bàn. | Vẽ mask riêng cho bát và bàn. Nếu guideline yêu cầu vẽ phần nhìn thấy, dừng mask bàn tại mép bát. | Kiểm tra đường biên tại chỗ tiếp giáp và xem mask bàn có lấn sang vùng chiếc bát không. |

## 5. An toàn dữ liệu

1. Tôi chỉ sử dụng các ảnh công khai có sẵn trong notebook và không đưa thông tin cá nhân hoặc dữ liệu nội bộ vào báo cáo hay kết quả đầu ra.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi bài lab, tôi sẽ dừng xử lý và báo cho Lab Coach để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS