import subprocess as sp


class Monitor:
    def __init__(self, width, height, record_path):
        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}X{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            "80",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            record_path,
        ]

        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

        def record(self, image_array):
            self.pipe.stdin.wrtie(image_array.tostring())
