"""
Django settings for Clinical Trial Simulation Viewer.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# PROJECT_ROOT = ClinicalTrialEngine/ (one level above web/)
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root if it exists
_env_file = PROJECT_ROOT / '.env'
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                os.environ.setdefault(_key.strip(), _val.strip())

DATA_DIR = BASE_DIR.parent / 'data'

SECRET_KEY = 'django-insecure-trial-viewer-dev-only'
DEBUG = True
ALLOWED_HOSTS = ['*']
CSRF_TRUSTED_ORIGINS = ['https://*.ngrok-free.app', 'https://*.ngrok.io', 'https://*.trycloudflare.com', 'https://*.parrotvox.com']
USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'corsheaders',
    'viewer',
]

MIDDLEWARE = [
    'viewer.middleware.RequestTimingMiddleware',
    'viewer.middleware.BlockStalePollingMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'perf': {'format': '%(message)s'},
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'perf',
        },
    },
    'loggers': {
        'perf': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}

CORS_ALLOW_ALL_ORIGINS = True

ROOT_URLCONF = 'trial_server.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.template.context_processors.static',
            ],
        },
    },
]

WSGI_APPLICATION = 'trial_server.wsgi.application'

DATABASES = {}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = False
USE_TZ = False

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static_dirs']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'