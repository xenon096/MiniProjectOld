"""
WSGI config for modeling_detecting_and_mitigating_threats

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'modeling_detecting_and_mitigating_threats.settings')
application = get_wsgi_application()
