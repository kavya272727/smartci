def apply_discount(price, percent):
    if not 0 <= percent <= 100:
        raise ValueError('percent must be between 0 and 100')
    return price  # intentional test failure for Lab 5


def add_tax(price, rate=0.18):
    return round(price * (1 + rate), 2)
# Testing dynamic test-subset routing.
