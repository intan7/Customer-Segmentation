	j�t�@j�t�@!j�t�@	B ̠�^@B ̠�^@!B ̠�^@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j�t�@��x�&1�?A\���(\�?Y��|?5^�?*	33333a@2F
Iterator::Model���QI��?!L��g�G@)_�Qګ?1?�ς�C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�q����?!n�\%�6@)�]K�=�?1xC��y3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��e�c]�?!d�7G4@)�z6�>�?1�x:,�0@:Preprocessing2U
Iterator::Model::ParallelMapV2/�$��?!l��(�@)/�$��?1l��(�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�Z�?!�`�,�>J@)M�O��?1Y6M�M�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!�[!�VH@){�G�zt?1�[!�VH@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!�����
@)HP�s�r?1�����
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz6�>W[�?!m�D�%�8@)��_vOf?1�/����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9A ̠�^@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��x�&1�?��x�&1�?!��x�&1�?      ��!       "      ��!       *      ��!       2	\���(\�?\���(\�?!\���(\�?:      ��!       B      ��!       J	��|?5^�?��|?5^�?!��|?5^�?R      ��!       Z	��|?5^�?��|?5^�?!��|?5^�?JCPU_ONLYYA ̠�^@b 