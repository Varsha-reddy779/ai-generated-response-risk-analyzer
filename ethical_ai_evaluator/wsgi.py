import os
from django.core.wsgi import get_wsgi_application

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ethical_ai_evaluator.settings')

application = get_wsgi_application()
