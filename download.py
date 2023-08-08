import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cems-fire-historical',
    {
        'product_type': 'reanalysis',
        'variable': [
            'build_up_index', 'danger_risk', 'drought_code',
            'duff_moisture_code', 'fine_fuel_moisture_code', 'fire_daily_severity_rating',
            'fire_weather_index',
        ],
        'version': '4.0',
        'dataset': 'Consolidated dataset',
        'year': [
            '2021', '2022',
        ],
        'month': [
            '03', '04', '05',
            '06', '07', '08',
            '09', '10',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'format': 'zip',
    },
    'download.zip')
