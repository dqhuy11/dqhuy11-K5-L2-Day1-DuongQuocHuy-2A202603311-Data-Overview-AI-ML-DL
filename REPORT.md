# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

Từ cần nhớ (đọc nhanh trước khi viết)
Từ		|Nghĩa ngắn
Prediction	|Kết quả model vừa đoán ra; chưa phải nhãn đúng
Ground truth	|Nhãn do người gán theo quy tắc
Taxonomy	|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)
Checkpoint	|File model .pt đã huấn luyện sẵn
Guideline	|Quy tắc gán nhãn do người viết
Annotator	|Người gán nhãn
Reviewer / QC	|Người kiểm tra nhãn
Escalation	|Ca khó → hỏi Lab Coach, không tự bịa
Model score	|Số “tin” của model; không phải điểm chất lượng nhãn người

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. -- Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K.
2. -- Record này nói về **cả bức ảnh**:Record chỉ có rank, class_name, score, không có tọa độ (bbox) hay polygon, nên không chỉ vào vật nào. Model nhìn toàn khung hình rồi xếp hạng các lớp, trong khi ảnh có rất nhiều xe buýt, ô tô và người. Đây là “đơn vị nhãn = cả ảnh”.
3.-- Class list từ đâu ra? Từ taxonomy đi kèm checkpoint. yolo11n-cls.pt đã được huấn luyện trên ImageNet-1K (1000 lớp cố định), nên nó chỉ chọn được trong danh sách đó. Model không “nghĩ ra” tên lớp mới. Vì vậy mới có chuyện ảnh giao thông mà top-5 toàn là loại xe (cab, minibus, police_van, recreational_vehicle, streetcar): không có lớp nào là “cảnh đường phố”.
4. -- Vì sao ghi cả class_id, class_name, taxonomy_name? Mỗi trường có một việc:
++ class_id là số để máy xử lý, ổn định.
++ class_name là tên để người đọc.
++ taxonomy_name cho biết số và tên đó thuộc bộ lớp nào.
Chỉ ghi tên lớp thì dễ nhầm, vì cùng một ID có thể là lớp khác ở taxonomy khác. Ví dụ trong lab: ID 468 là cab ở ImageNet-1K, nhưng ở detection (COCO-80) ID 5 là bus và ID 2 là car.
5.-- Cần nói rõ tiêu chí chọn đúng một lớp, ví dụ chọn vật chiếm diện tích lớn nhất, chủ đề chính của cảnh, hoặc thứ tự ưu tiên giữa các lớp. Cũng cần nói khi nào không lớp nào khớp thì làm gì (hỏi lại, hay dùng nhãn “khác”). Ảnh traffic có xe buýt xanh, ô tô, người đi bộ. Không có quy tắc thì annotator A chọn “bus”, annotator B chọn “car”, và nhãn không thống nhất. 
6. -- Vì score là độ tin của model, không do người kiểm tra theo guideline. Ground truth là nhãn do người gán theo quy tắc, rồi được reviewer kiểm tra. Ví dụ trong ảnh này, model tự tin 0.51 là cab trong khi ảnh chủ yếu là xe buýt và ô tô. Model có thể tự tin mà vẫn sai, hoặc đúng mà score thấp. Prediction sai không làm bài kém đi, mà còn là evidence tốt.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1.-- Chọn Record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. ++ xyxy nghĩa là [x_min, y_min, x_max, y_max], đơn vị pixel, gốc (0, 0) ở góc trên-trái ảnh. x tăng sang phải, y tăng xuống dưới.
++ Góc trên-trái của hộp ở (385.33, 69.24): cách mép trái ảnh khoảng 385 px, cách mép trên khoảng 69 px.
++ Góc dưới-phải ở (498.92, 348.92).
++ Rộng = 498.92 − 385.33 ≈ 113.6 px (JSON ghi 113.58 vì làm tròn từ số gốc). Cao = 348.92 − 69.24 = 279.68 px.
++ Hộp cao gần 2,5 lần bề rộng, khớp với người đàn ông đứng quay lưng ở bên phải ảnh.
3.-- Số prediction theo ngưỡng:
++ 0.20 → 17 prediction.
++ 0.35 → 11 prediction (ngưỡng dùng cho JSON).
++ 0.60 → 6 prediction.
-- score_threshold (ngưỡng conf) là mức tối thiểu để model giữ lại một prediction. Hạ ngưỡng thì giữ nhiều hộp hơn, nâng ngưỡng thì loại bớt. 
4. -- Hạ ngưỡng thì reviewer phải làm thêm gì? Phải xem thêm nhiều hộp hơn, và các hộp thêm đều có score thấp nên dễ sai. So 0.20 với 0.35, thêm 6 hộp (spoon ×3, potted plant, dining table, bottle). Với mỗi hộp, reviewer kiểm tra lớp có đúng không, hộp có sát vật không, có hộp thừa hay trùng không. Ngược lại, nâng ngưỡng lên 0.60 thì mất 5 vật thật vẫn thấy trong ảnh (3 bowl, 2 cup). Nên ngưỡng chỉ lọc prediction, không phải quy tắc “vật nào cần gán nhãn”.
5. -- Quy tắc box (áp dụng cho nhiều ảnh): Mỗi vật thể nhìn thấy được có đúng một hộp ôm sát phần nhìn thấy của vật. Không để dư nền, không gộp hai vật tách rời vào một hộp. Vật bị che hoặc bị cắt mép ảnh thì chỉ ôm phần nhìn thấy. Quy tắc tốt là quy tắc không nhắc tên ảnh hay vật cụ thể.
6. --Vật bị che/cắt mép và chọn guideline hay escalation: Trên kitchen, có một person bị cắt ở mép trái ảnh (score 0.610931, box [0.12, 263.18, 61.55, 310.78]), chỉ thấy cánh tay thò vào khung hình.
++ Cách vẽ box cho vật bị cắt mép (chỉ ôm phần nhìn thấy) là guideline, vì áp dụng được cho mọi ảnh.
++ Còn phần nhìn thấy nhỏ đến mức nào thì vẫn gán person (chỉ một cánh tay có tính không?) guideline chưa quy định. Đó là ca escalation: hỏi Lab Coach, không tự quyết.
-- Nguyên tắc chung: ca nào quy tắc đã phủ thì làm theo guideline, ca nào không đủ thông tin hoặc guideline chưa nói thì hỏi, không tự bịa.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. -- Record chọn: instance_id=kitchen-001, class_name=person, score=0.899318, polygon_point_count=348, 5 điểm đầu của polygon_xy = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]. Mỗi điểm là một cặp [x, y] theo pixel, nối lại theo thứ tự thành đường viền của vật.
2.-- Polygon cho thêm gì so với box? Box chỉ là hình chữ nhật 4 số (ở đây [385.45, 66.44, 498.02, 348.58]) nên chứa cả nền xung quanh người. Polygon 348 điểm bám theo viền thật: đầu, vai, tay chống hông, thân, hai chân. Nhờ đó biết được hình dạng và phần nền nằm trong hộp nhưng không thuộc người (ví dụ khoảng trống giữa hai chân).
3.-- instance_id dùng để làm gì? Để phân biệt từng vật riêng trong output của lab. Trong kitchen, kitchen-002, -003, -006, -010 đều là bowl nhưng là bốn instance khác nhau, mỗi cái một polygon. Câu “không phải class ID và không phải tracking ID” nghĩa là:
++ Class ID là mã loại vật (như person=0, bus=5). Nhiều vật cùng loại dùng chung một class ID, còn instance_id thì mỗi vật một mã.
++ Tracking ID theo dõi cùng một vật qua nhiều khung hình video. Đây chỉ là một ảnh tĩnh, kitchen-001 chỉ là số thứ tự trong output này, không có nghĩa gì ngoài file đó.
4. -- Quy tắc vẽ biên mask: Polygon bám sát đường viền phần nhìn thấy của vật, không lẫn nền hay vật khác. Vùng bị vật đứng trước che thì thuộc vật đứng trước. Hai vật cùng lớp chạm nhau thì vẽ hai polygon riêng.
5. -- Chỗ khó trên kitchen: Góc dưới-trái có mask oven (kitchen-007), dining table (kitchen-005), person (kitchen-009) và nhiều bowl chồng lên nhau; mặt bàn còn bị bát, lọ che từng phần.
++ Cách xử lý vật chồng nhau (“pixel thuộc vật đứng trước”) là guideline.
++ Vùng tối phía sau bàn có phải oven hay không (nhìn giống bếp lò có lửa hơn) là câu guideline chưa phủ, nên escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ xem prediction. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_name` (kèm `class_id`, `taxonomy_name`) cho cả ảnh | Ảnh `traffic` có nhiều xe buýt và ô tô nhưng top-1 là `cab` (0.51), tiếp theo là `minibus` (0.16) và `police_van` (0.09): các lớp xe gần nhau và không có lớp "cảnh giao thông" | Chọn một lớp theo tiêu chí chủ thể chính trong guideline; ca không rõ thì escalation | Kiểm nhãn có đúng quy tắc chọn không, có nhất quán giữa các ảnh không; không dựa vào score |
| Phát hiện vật thể | Mỗi vật một hộp `[x_min, y_min, x_max, y_max]` (pixel) kèm lớp | Ảnh `kitchen`: `person 0.61` bị cắt ở mép trái; các `bowl` chồng lên nhau; số hộp đổi từ 17 xuống 6 khi đổi ngưỡng từ 0.20 lên 0.60 | Vẽ hộp sát cho từng vật nhìn thấy, gán lớp, không bỏ vật dù model score thấp | Kiểm hộp có sát không, đúng lớp không, có thiếu/thừa hộp không, cách xử lý vật bị che/cắt |
| Instance segmentation | Mỗi đối tượng một polygon `polygon_xy` kèm lớp và `instance_id` riêng | Ảnh `kitchen`: mask `dining table 0.58` chồng lên bát và bột; `oven 0.49`, `person 0.43` chạm nhau; bó lá khô bị gán `potted plant 0.63` | Vẽ biên theo viền nhìn thấy, tách từng instance, hai vật cùng lớp có polygon riêng | Kiểm biên mask có bám vật không, instance có bị gộp hoặc tách nhầm không, lớp có đúng không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
-- Chỉ dùng ba ảnh COCO công khai có ghi nguồn và giấy phép (CC BY 2.0), không tải ảnh cá nhân hay dữ liệu nội bộ, và không ghi họ tên, MSSV hay dữ liệu cá nhân vào báo cáo hoặc output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   Dừng thao tác và báo Lab Coach / GV phụ trách lab.

## 6. Checklist trước khi nộp link repo
Kiểm tra trên GitHub trong thư mục report/:

[X] REPORT.md (file này đã điền)
[X] day1_lab_outputs/classification_predictions.json
[X] day1_lab_outputs/detection_predictions.json
[X] day1_lab_outputs/segmentation_predictions.json
[X] day1_lab_outputs/IMAGE_ATTRIBUTION.md
[X] day1_lab_outputs/visuals/classification_top5.png
[X] day1_lab_outputs/visuals/detection_predictions.png
[X] day1_lab_outputs/visuals/segmentation_prediction.png
[X] Ô validation cuối notebook báo PASS
[X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
[X] Đã nộp link repository trên VLearn
