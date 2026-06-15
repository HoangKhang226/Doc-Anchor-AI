"""
Doc Anchor AI — VLM OCR
Trích xuất thông tin end-to-end từ hình ảnh chụp (đã qua OpenCV) 
Sử dụng Qwen2-VL thông qua Ollama.

Hỗ trợ Dynamic Temperature: nhiệt độ VLM được điều chỉnh tự động
dựa trên LayoutRouter (bảng biểu = 0.0, text thuần = 0.2).
"""

from pathlib import Path
import base64

from langchain_core.messages import HumanMessage
from src.core.llm_factory import get_vlm
from src.config import get_logger

logger = get_logger(__name__)


class VLMOCRProcessor:
    """Class bóc tách nội dung ảnh bằng Vision Language Model.
    
    Hỗ trợ Dynamic Temperature: temperature có thể được truyền vào
    hàm extract() theo từng file thay vì cố định lúc khởi tạo.
    """
    
    def __init__(self, temperature: float = 0.0):
        self._default_temperature = temperature
        self.vlm = get_vlm(temperature=temperature)
        self.default_prompt = (
            "You are a multilingual document OCR system. "
            "Extract all content faithfully, preserve the original language, "
            "construct valid Markdown tables, and never hallucinate data."
        )

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """Chuyển ảnh sang base64 và xác định mime type. Tự động convert WebP sang JPEG cho Ollama."""
        from PIL import Image
        import io
        
        ext = image_path.suffix.lower()
        if ext == '.webp':
            # Ollama Vision không hỗ trợ tốt WebP, nên convert sang JPEG
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return "image/jpeg", base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_path, "rb") as image_file:
            return mime_type, base64.b64encode(image_file.read()).decode('utf-8')

    def _get_vlm_with_temperature(self, temperature: float):
        """Lấy VLM instance với temperature phù hợp. Tái sử dụng nếu cùng temperature."""
        if temperature == self._default_temperature:
            return self.vlm
        logger.info(f"Dynamic Temperature: khởi tạo VLM với temperature={temperature}")
        return get_vlm(temperature=temperature)

    def extract(
        self,
        image_path: str | Path,
        system_prompt: str | None = None,
        ocr_blocks: list | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Sử dụng VLM local để đọc nội dung ảnh.
        
        Args:
            image_path: Đường dẫn đến file ảnh.
            system_prompt: Prompt tùy chỉnh (nếu có).
            ocr_blocks: Danh sách OCR blocks từ PaddleOCR (context cho VLM).
            temperature: Nhiệt độ VLM động từ LayoutRouter. 
                         None = dùng default (0.0).
                         0.0 = bám sát OCR (bảng/scan).
                         0.2 = hành văn mượt (text thuần).
        """
        path = Path(image_path)
        target_temp = temperature if temperature is not None else self._default_temperature
        logger.info(f"Đang bóc tách ảnh bằng VLM: {path.name} (temperature={target_temp})")
        
        mime_type, base64_image = self._encode_image(path)
        prompt = system_prompt or self.default_prompt
        
        # Lấy VLM với temperature phù hợp
        vlm = self._get_vlm_with_temperature(target_temp)
        
        # Tạo message theo chuẩn LangChain cho multi-modal
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )
        
        try:
            response = vlm.invoke([message])
            logger.info("VLM bóc tách thành công.")
            return response.content
        except Exception as e:
            logger.error(f"Lỗi khi gọi VLM: {e}")
            raise RuntimeError(f"VLM OCR failed: {e}")

