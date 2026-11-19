import requests
import json
import sys

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    url = "http://127.0.0.1:5000/evaluate"

    print("=" * 60)
    print("  IELTS WRITING TASK 2 INTERACTIVE CLIENT")
    print("=" * 60)

    # 1. Nhập đề bài
    print("\n[1] Nhập đề bài IELTS Writing Task 2:")
    prompt = input("> ").strip()
    if not prompt:
        print("Lỗi: Đề bài không được để trống.")
        return

    # 2. Nhập bài viết
    print("\n[2] Nhập bài viết của bạn (gõ 'DONE' trên một dòng riêng để kết thúc):")
    essay_lines = []
    while True:
        line = input()
        if line.strip().upper() == "DONE":
            break
        essay_lines.append(line)
    
    essay = "\n".join(essay_lines).strip()
    if not essay or len(essay.split()) < 10:
        print("Lỗi: Bài viết quá ngắn (phải tối thiểu 10 từ).")
        return

    print("\nĐang gửi bài viết tới API server để chấm điểm...")
    payload = {
        "prompt": prompt,
        "essay": essay,
        "debug": False  # Đổi thành True nếu bạn muốn xem chi tiết log pipeline đặc trưng
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"\n❌ Lỗi từ server ({response.status_code}): {response.json().get('error')}")
            return
        
        res = response.json()
        
        print("\n" + "=" * 60)
        print("🎉 KẾT QUẢ CHẤM ĐIỂM IELTS WRITING V2")
        print("=" * 60)
        print(f"👉 OVERALL BAND SCORE: {res.get('band')} / 9.0")
        print("-" * 60)
        print(f"   - Task Response (TR):          {res.get('task_response')}")
        print(f"   - Coherence & Cohesion (CC):   {res.get('coherence')}")
        print(f"   - Lexical Resource (LR):       {res.get('lexical')}")
        print(f"   - Grammatical Accuracy (GRA):  {res.get('grammar')}")
        print("-" * 60)
        
        # In cảnh báo độ tin cậy thấp (nếu có)
        if "warning" in res:
            print(f"⚠️ CẢNH BÁO: {res.get('warning')}\n")

        feedback = res.get("feedback", {})
        
        # 3. Điểm mạnh
        print("💪 ĐIỂM MẠNH (STRENGTHS):")
        for s in feedback.get("strengths", []):
            print(f"  • {s}")
        
        # 4. Điểm cần cải thiện
        print("\n📈 ĐIỂM CẦN CẢI THIỆN (WEAKNESSES):")
        for imp in feedback.get("improvements", []):
            print(f"  • {imp}")
            
        # 5. Các ưu tiên cần học
        print("\n🎯 THỨ TỰ ƯU TIÊN ÔN TẬP (TUTOR PRIORITIES):")
        for priority in feedback.get("priorities", []):
            print(f"  Priority {priority.get('priority_rank')}: {priority.get('criterion_name')}")
            print(f"    - Thiếu hụt (Deficit): -{priority.get('deficit'):.2f}")
            print(f"    - Gợi ý: {priority.get('recommendation')}")
        
        # 6. Bản sửa lỗi ngữ pháp & cấu trúc
        print("\n" + "=" * 60)
        print("✍️ BẢN SỬA LỖI & ĐỀ XUẤT NÂNG CẤP (REWRITE)")
        print("=" * 60)
        rewrite_data = res.get("rewrite", {})
        
        # Danh sách lỗi ngữ pháp
        corrections = rewrite_data.get("corrections_made", [])
        if corrections:
            print("🔍 Các lỗi ngữ pháp đã được tự động sửa:")
            # Lấy tối đa 10 lỗi để tránh tràn màn hình
            for corr in corrections[:10]:
                if corr.get('original'):
                    print(f"  • Sửa \"{corr.get('original')}\" thành \"{corr.get('corrected')}\" -> {corr.get('message')}")
                else:
                    # Trường hợp LanguageTool bị tắt
                    print(f"  • {corr.get('message')}")
            if len(corrections) > 10:
                print(f"  ... và {len(corrections) - 10} lỗi khác.")
        else:
            print("🔍 Không phát hiện lỗi ngữ pháp nào cần sửa.")

        print("\n📝 Đề xuất cải thiện cấu trúc bài viết (Suggestions):")
        print(rewrite_data.get("structural_suggestions"))

        # Thời gian xử lý
        timing = res.get("timing_ms", {})
        print("\n" + "-" * 60)
        print(f"⏱️ Tổng thời gian xử lý: {timing.get('total_ms', 0):.1f}ms (Trích xuất: {timing.get('feature_extraction_ms', 0):.1f}ms | Dự đoán: {timing.get('ml_prediction_ms', 0):.1f}ms)")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Lỗi kết nối tới API Server: {e}")
        print("Hãy chắc chắn rằng server Flask 'app.py' đang chạy trên Terminal 1.")

if __name__ == "__main__":
    main()
