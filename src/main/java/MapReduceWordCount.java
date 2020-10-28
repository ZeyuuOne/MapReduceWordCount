import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class MapReduceWordCount {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        enum CountersEnum {INPUT_WORDS}

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private Configuration conf = new Configuration();
        private Boolean caseSensitive;
        private Set<String> patternsToSkip = new HashSet<String>();
        private Set<String> stopWords = new HashSet<String>();

        private void parseSkipFile(String fileName) throws FileNotFoundException {
            try {
                BufferedReader fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) {
                    patternsToSkip.add(pattern);
                }
            } catch (IOException e) {
                System.err.println("Caught exception while parsing the cached file " + StringUtils.stringifyException(e));
            }
        }

        private void parseStopWordsFile(String fileName) throws FileNotFoundException {
            try {
                BufferedReader fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) {
                    stopWords.add(pattern);
                }
            } catch (IOException e) {
                System.err.println("Caught exception while parsing the cached file " + StringUtils.stringifyException(e));
            }
        }

        private boolean isNumber(String str) {
            for (int i = 0; i < str.length(); i++) {
                if (!Character.isDigit(str.charAt(i))) return false;
            }
            return true;
        }

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            conf = context.getConfiguration();
            caseSensitive = conf.getBoolean("wordcount.case.sensitive", true);
            URI[] URIs = Job.getInstance(conf).getCacheFiles();

            if (-1 < conf.getInt("wordcount.skip.patterns", -1)) {
                Path patternsPath = new Path(URIs[conf.getInt("wordcount.skip.patterns", -1)].getPath());
                String patternsFileName = patternsPath.getName().toString();
                parseSkipFile(patternsFileName);
            }
            if (-1 < conf.getInt("wordcount.stop.words", -1)) {
                Path stopWordsPath = new Path(URIs[conf.getInt("wordcount.stop.words", -1)].getPath());
                String stopWordsFileName = stopWordsPath.getName().toString();
                parseStopWordsFile(stopWordsFileName);
            }

        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
            for (String pattern : patternsToSkip) {
                line = line.replaceAll(pattern, " ");
            }
            StringTokenizer itr = new StringTokenizer(line);
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                if (token.length() < 3 || isNumber(token)) continue;
                boolean stop = false;
                for (String stopWord : stopWords) {
                    if (stopWord.equals(token)) {
                        stop = true;
                        break;
                    }
                }
                if (stop) continue;
                word.set(token);
                context.write(word, one);
                Counter counter = context.getCounter(CountersEnum.class.getName(), CountersEnum.INPUT_WORDS.toString());
            }
        }

    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class IntWritableDecreaseComparator extends IntWritable.Comparator {
        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            return -super.compare(a, b);
        }

        @Override
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static class OrderRecordWriter extends RecordWriter<IntWritable,Text>{

        FSDataOutputStream fos = null;
        Integer order = 0;

        public OrderRecordWriter(TaskAttemptContext job){
            FileSystem fs;
            try {
                fs = FileSystem.get(job.getConfiguration());
                Path outputPath = new Path("output/out.txt");
                fos = fs.create(outputPath);
            } catch (IOException e){
                System.err.println("Caught exception while getting the configuration " + StringUtils.stringifyException(e));
            }
        }

        @Override
        public void write(IntWritable key, Text value) throws IOException, InterruptedException {
            if (order > 99) return;
            fos.write(((++order).toString()+": "+value.toString()+", "+key.toString()+"\n").getBytes());
        }

        @Override
        public void close(TaskAttemptContext context) throws IOException, InterruptedException {
            IOUtils.closeStream(fos);
        }
    }

    public static class OrderOutputFormat extends FileOutputFormat<IntWritable, Text> {
        @Override
        public RecordWriter<IntWritable, Text> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
            return new OrderRecordWriter(job);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();

        Path tempPath = new Path("temp");

        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(MapReduceWordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        List<String> otherArgs = new ArrayList<String>();
        int cacheIndex = -1;
        for (int i = 0; i < remainingArgs.length; i++) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setInt("wordcount.skip.patterns", ++cacheIndex);
            } else if ("-stop".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setInt("wordcount.stop.words", ++cacheIndex);
            } else {
                otherArgs.add(remainingArgs[i]);
            }
        }

        FileInputFormat.addInputPath(job, new Path(otherArgs.get(0)));
        FileOutputFormat.setOutputPath(job, tempPath);

        if (!job.waitForCompletion(true)) System.exit(1);

        Job sortJob = Job.getInstance(conf, "value sort");
        sortJob.setJarByClass(MapReduceWordCount.class);
        sortJob.setInputFormatClass(SequenceFileInputFormat.class);
        sortJob.setMapperClass(InverseMapper.class);
        sortJob.setNumReduceTasks(1);
        sortJob.setOutputKeyClass(IntWritable.class);
        sortJob.setOutputValueClass(Text.class);
        sortJob.setSortComparatorClass(IntWritableDecreaseComparator.class);
        sortJob.setOutputFormatClass(OrderOutputFormat.class);

        FileInputFormat.addInputPath(sortJob, tempPath);
        FileOutputFormat.setOutputPath(sortJob, new Path(otherArgs.get(1)));

        if (!sortJob.waitForCompletion(true)) System.exit(1);

        System.exit(0);

    }
}