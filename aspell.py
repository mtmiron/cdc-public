import logging
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Aspell:
    def __init__(self):
        self.aspell = self._spawn()

    def _spawn(self):
        aspell = subprocess.Popen(["aspell", "-a"], stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       #stderr=subprocess.STDOUT,
                                       bufsize=1,
                                       #restore_signals=False,
                                       encoding="iso-8859-1")
                                       #universal_newlines=True)
        aspell.stdout.readline()
        return aspell

    def _recycle(self):
        if self.aspell.poll() is not None:
            self.aspell.terminate()
            del self.aspell
            self.aspell = self._spawn()

    def _write(self, text):
        return self.aspell.stdin.write( ("^" + text.rstrip() + "\n") )

    def _readline(self):
        return self.aspell.stdout.readline()
   
    def _parse_correction(self, response):
        first, second = response[2:].split(":")
        orig, count, offset = first.split(" ")
        correction = second.split(",")[0][1:]
        return orig, count, offset, correction

    def _replace(self, string, response):
        orig, count, offset, correction = self._parse_correction(response)
        return self.replace_by_slice(string, slice(int(offset), int(offset) + len(orig)), correction)

    def replace_by_slice(self, string, slyce, sub):
        return string[:slyce.start - 1] + sub + string[slyce.stop - 1:]

    def correct(self, text):
        try:
            self._write(text)
        except BrokenPipeError:
            self._recycle()
            self._write(text)

        while True:
            response = self._readline()
            if response == "\n": break  # finished with this sentence
            elif response[0] == "*": continue  # no spelling error for the current word
            elif response[0] == "&": text = self._replace(text, response)  # suggestions available to correct current word
            elif response[0] == "#": continue  # no suggestions, but the word is not recognized
            else: logger.error("unknown response, received: {}".format(response))
        return text

if __name__ == "__main__":
    import preproc
    import pandas as pd
    speller = Aspell()
    df = pd.read_csv("valid5000.csv")
    for i in range(5):
        print(df['text'][i])
    df = preproc.preproc(df, lower=True, spelling=True)
    for i in range(5):
        print(df['text'][i])
