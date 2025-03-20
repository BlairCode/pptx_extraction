import os
import time
import ffmpeg
from datetime import timedelta
from faster_whisper import WhisperModel
import opencc

class SubtitleGenerator:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        """初始化字幕生成器"""
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.converter = opencc.OpenCC('t2s')
        self.model = None
        
    def load_model(self):
        """加载语音识别模型"""
        try:
            print(f"加载 {self.model_size} 模型（{self.device} 模式）...")
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            return True
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            return False

    def generate_subtitles(self, audio_file, language="zh", beam_size=5):
        """生成字幕文本"""
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"音频文件 {audio_file} 不存在")
            
            if not self.model:
                self.load_model()
                
            print("正在识别...")
            segments, _ = self.model.transcribe(
                audio_file, 
                language=language, 
                beam_size=beam_size, 
                word_timestamps=True
            )
            
            subtitles = ""
            segment_id = 1
            
            for segment in segments:
                text = self.converter.convert(segment.text.strip())
                sentences = text.split("。")
                if not sentences[-1]:
                    sentences.pop()
                
                duration = segment.end - segment.start
                sentence_duration = duration / len(sentences) if sentences else duration
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        start_time = segment.start + i * sentence_duration
                        end_time = start_time + sentence_duration
                        subtitles += (
                            f"{segment_id}\n"
                            f"{self._format_timestamp(start_time)} --> "
                            f"{self._format_timestamp(end_time)}\n"
                            f"{sentence.strip()}\n\n"
                        )
                        segment_id += 1
            
            return subtitles
            
        except FileNotFoundError as e:
            return f"错误：{str(e)}"
        except Exception as e:
            return f"发生错误：{str(e)}"

    def save_to_srt(self, subtitle_text, filename="output.srt"):
        """保存字幕到SRT文件"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(subtitle_text)
            return True
        except Exception as e:
            print(f"保存字幕失败：{str(e)}")
            return False

    # def generate_video_with_subtitles(self, audio_file, subtitle_file, output_file="output_with_subtitles.mp4"):
    #     """生成带字幕的视频"""
    #     try:
    #         command = (
    #             f'ffmpeg -f lavfi -i "color=c=black:s=1280x720" '
    #             f'-i "{audio_file}" '
    #             f'-vf "subtitles={subtitle_file}" '
    #             f'-c:a aac -shortest "{output_file}"'
    #         )
    #         os.system(command)
    #         return True
    #     except Exception as e:
    #         print(f"视频生成失败：{str(e)}")
    #         return False

    @staticmethod
    def _format_timestamp(seconds):
        """格式化时间戳"""
        ms = int((seconds % 1) * 1000)
        time_str = str(timedelta(seconds=int(seconds)))
        if len(time_str) < 8:
            time_str = "0" + time_str
        return f"{time_str},{ms:03d}"

# 使用示例
if __name__ == "__main__":
    generator = SubtitleGenerator(model_size="medium", device="cpu")
    audio_file = "G:/我的云端硬盘/PPTX/PPT_Text_Extractor/sample/sample.mp3"
    subtitles = generator.generate_subtitles(audio_file, language="zh")
    print("字幕结果：")
    print(subtitles)
    generator.save_to_srt(subtitles, "output.srt")
    print("字幕已保存到 output.srt")
    # generator.generate_video_with_subtitles(audio_file, "output.srt")
    # print("已生成 output_with_subtitles.mp4")