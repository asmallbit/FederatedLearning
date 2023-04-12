import requests
import json

def send_telegram_message(message: str,
                          user_id: str,
                          api_key: str,
                          proxy_type: str = None,
                          proxy_host: str = None,
                          proxy_port: str = None,
                          proxy_username: str = None,
                          proxy_password: str = None):

    proxies = None
    if proxy_type is not None and proxy_host is not None:
        if proxy_type not in ["socks5", "http", "https"]:
            print("proxy type %s unknown, disable proxy", proxy_type)
            raise Exception()

        proxies = {
            'https': f'{proxy_type}://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}',
            'http': f'{proxy_type}://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}'
        }
    headers = {'Content-Type': 'application/json',
                'Proxy-Authorization': 'Basic base64'}
    data_dict = {'chat_id': user_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_notification': True}
    data = json.dumps(data_dict)
    url = f'https://api.telegram.org/bot{api_key}/sendMessage'
    response = requests.post(url,
                                data=data,
                                headers=headers,
                                proxies=proxies,
                                verify=True)
    return response