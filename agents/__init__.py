import dishes
import pyramid
import water


def catalog(config):
    return {
        'dishes': dishes.DishesAgent,
        'pyramid': pyramid.PyramidAgent,
        'water': water.WaterAgent,
    }[config.domain_name](config)
