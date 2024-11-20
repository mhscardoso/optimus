from manim import *

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        self.play(Create(circle))


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)

        text = MathTex(r"""
f(x + \delta) = 3(x + \delta)^2 + 9\delta
""")

        square = Square()

        self.play(Write(text))
        self.wait(1.5)
        self.play(square.animate.rotate(-PI / 4))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
