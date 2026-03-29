from django.db import models

class Evaluation(models.Model):

    prompt = models.TextField()

    response = models.TextField()

    evaluation = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prompt