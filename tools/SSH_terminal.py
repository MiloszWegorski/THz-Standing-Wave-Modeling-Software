import paramiko
import threading
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QFont, QColor, QPalette, QKeyEvent, QTextCursor
from PySide6.QtCore import Signal, QObject
import re

ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class SSHSig(QObject):
    recieved_sig = Signal(str)

class SSHTerminal(QTextEdit):
    def __init__(self, hostname, username, password):
        super().__init__()

        #text formatting
        self.setFont(QFont("Consolas", 11))

        palette=self.palette()
        palette.setColor(QPalette.Base, QColor("#1e1e1e"))
        palette.setColor(QPalette.Text, QColor("#00ff00"))
        self.setPalette(palette)

        #set as editable
        # self.setReadOnly(False)
        
        self.signal = SSHSig()

        self.signal.recieved_sig.connect(self.append_output)

        #establish SSH connection
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.channel = None
        threading.Thread(target=self._connect_ssh, 
                        args=(hostname, username, password), daemon=True).start()


    def append_output(self, rec_sig):
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(rec_sig)
        self.moveCursor(QTextCursor.MoveOperation.End)

    def _connect_ssh(self, hostname, username, password):
        try:
            self.append(f"Connecting to {hostname}...\n")

            # connecting to ssh server 
            self.client.connect(hostname=hostname, username=username, password=password)
            self.channel = self.client.invoke_shell(term='xterm')
            
            self.signal.recieved_sig.emit(f"Connected to {hostname}\n")

            #start reading SSh output continously
            threading.Thread(target=self._read_channel, daemon=True).start()
        
        except Exception as e:
            self.signal.recieved_sig.emit(f"Connection error: {e}\n")

    def _read_channel(self):
        while True:
            if self.channel and self.channel.recv_ready():
                data = self.channel.recv(1024).decode(errors="ignore")
                self.signal.recieved_sig.emit(data)

    def _append_output(self, text):
        text = ANSI_ESCAPE_RE.sub('', text)
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)


    def keyPressEvent(self, event: QKeyEvent):
        if not self.channel:
            return

        key = event.key()
        text = event.text()

        # ENTER
        if key == 16777220:  # Qt.Key_Return / Enter
            self.channel.send("\r")

        # # BACKSPACE
        # elif key == 16777219:  # Qt.Key_Backspace
        #     self.channel.send("\x7f")  # Send DEL

        # TAB
        elif key == 16777218:  # Qt.Key_Tab
            self.channel.send("\t")

        # ARROWS (optional — only for shells that support it)
        elif key == 16777235:  # Up
            self.channel.send("\x1b[A")
        elif key == 16777237:  # Down
            self.channel.send("\x1b[B")
        elif key == 16777234:  # Left
            self.channel.send("\x1b[D")
        elif key == 16777236:  # Right
            self.channel.send("\x1b[C")

        # NORMAL CHARACTERS
        elif text:
            self.channel.send(text)