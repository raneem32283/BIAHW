# Generated by Django 5.0.6 on 2024-06-17 13:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='city',
            old_name='CityName',
            new_name='name',
        ),
        migrations.AddField(
            model_name='city',
            name='x',
            field=models.DecimalField(decimal_places=10, default=0, max_digits=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='city',
            name='y',
            field=models.DecimalField(decimal_places=10, default=0, max_digits=10),
            preserve_default=False,
        ),
    ]
