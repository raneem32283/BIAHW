from django.urls import path
from . import views
urlpatterns = [

    path('',views.distance, name='distance'),
    path('',views.initialize_population, name='initialize_population'),
    path('',views.fitness, name='fitness'),
    path('',views.selection, name='selection'),
    path('',views.crossover, name='crossover'),
    path('',views.mutate, name='mutate'),
    path('',views.genetic_algorithm, name='genetic_algorithm'),
    path('',views.tsp, name='tsp'),
    path('',views.calculate_optimal_routes_for_cars, name='calculate_optimal_routes_for_cars'),
    path('',views.your_view, name='your_view'),
    path('',views.your_endpoint, name='your_endpoint'),
]