import dishes
import pyramid
import water


def catalog(name):
    return {
        'dishes': dishes.DishesEnv,
        'pyramid': pyramid.PyramidEnv,
        'water': water.WaterEnv
    }[name]()
