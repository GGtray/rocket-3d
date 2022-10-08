class functons_as_objects:
    def __init__(self) -> None:
        print("wow, inited")
        pass
        
    def __call__(self, input):
        print("wow, functons as objects" + input)


if __name__ == '__main__':
    fun_as_ob = functons_as_objects()
    fun_as_ob('fu')