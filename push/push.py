from push.telegram.telegram_push import *

class Push():
    def __init__(self, conf):
        self.conf = conf

    def push(self, message):
        if self.conf["is_push_enable"]:
            # 使用Telegram bot push
            if self.conf["push_type"].lower() == "Telegram".lower():
                send_telegram_message(message, 
                            user_id = self.conf["telegram_id"],
							api_key = self.conf["api_key"],
							proxy_type = self.conf["proxy_type"],
							proxy_host = self.conf["proxy_host"],
							proxy_port = self.conf["proxy_port"],
							proxy_username = self.conf["proxy_username"],
							proxy_password = self.conf["proxy_password"])