from django.db import models


class FlightInput(models.Model):
    # Assuming 3-letter airport code
    departure_city = models.CharField(max_length=10)
    destination_city = models.CharField(max_length=10)
    date = models.DateField()
    adults_count = models.CharField(max_length=20)
    children_count = models.CharField(max_length=20)
    infants_count = models.CharField(max_length=20) 
