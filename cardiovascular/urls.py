from django.urls import include, path

from api import views

from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('',views.index,name='api Documentation'),

    # path('api/v2/twitter/sentiment', views.cardiorisk2),
    path('api/v2/twitter/sentiment', views.twitterSentiment),
    path('api/v2/news/sentiment', views.newsAnanlyserView),
    path('api/v2/all/stats', views.allStats),
    path('api/v2/twitter/download', views.save_file),
    path('admin/', admin.site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)