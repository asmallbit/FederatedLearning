from push.push import Push

def notify_user(message: str, push: Push):
	print(message)
	push.push(message)