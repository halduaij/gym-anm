.. _offline_training:

Offline training utilities
==========================

The package provides basic utilities to collect datasets of state--action pairs
which can later be used for behaviour cloning or other offline RL methods.

Generating datasets
-------------------

Datasets can be created from a mix of agents with different levels of
expertise.  The function :func:`generate_mixed_dataset` selects an agent from a
given list at each step and records the resulting transition.  Agents can be
drawn uniformly or according to user-specified sampling ``weights``.

Several simple experts are provided.  In addition to conservative and
aggressive variants, ``NoisyCapBankExpert`` simulates measurement errors while
``DelayedCapBankExpert`` acts only every few steps. ``StaleCapBankExpert`` makes
decisions using delayed observations.  These allow generating more realistic
sub-optimal behaviour.

.. literalinclude:: ../../../examples/offline_mixed.py
   :language: python
   :start-after: BEGIN OFFLINE MIXED EXAMPLE
   :end-before: END OFFLINE MIXED EXAMPLE
