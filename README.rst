|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Гибридный подход распознавания рукописного текста
    :Тип научной работы: M1P
    :Автор: Руслан Александрович Войт
    :Научный руководитель: д.т.н., профессор Местецкий Леонид Моисеевич

Abstract
========

В статье исследуется гибридный подход к распознаванию рукописного текста, сочета-
ющий визуальный анализ растрового изображения и структурный анализ векторного
графа штрихов. Существующие растровые модели требуют больших объемов размечен-
ных выборок данных и не учитывают геометрию письма, что снижает их эффективность
на исторических документах с высокой вариативностью почерков. Для решения этой
проблемы предложена двухпоточная архитектура: визуальный поток на базе Vertical
Attention Network дополнен графовым потоком, где граф штрихового разложения стро-
ится методом непрерывной скелетизации на основе диаграммы Вороного, а его признаки
обрабатываются графовой нейронной сетью с последующим слиянием модальностей и
CTC-декодированием. Эксперименты на данных из датасета IAM подтвердили устойчи-
вость предложенного графового представления, продемонстрировав снижение функции
потерь на тестовой выборке относительно растровых аналогов. Результаты применимы
для оцифровки исторических рукописей и архивных документов.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
