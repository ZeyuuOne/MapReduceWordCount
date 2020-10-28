import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
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

        private boolean isNumber(String str){
            for (int i = 0; i < str.length(); i++){
                if (!Character.isDigit(str.charAt(i))) return false;
            }
            return true;
        }

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            conf = context.getConfiguration();
            caseSensitive = conf.getBoolean("wordcount.case.sensitive", true);
            URI[] URIs = Job.getInstance(conf).getCacheFiles();

            if (conf.getBoolean("wordcount.skip.patterns", true)) {
                Path patternsPath = new Path(URIs[0].getPath());
                String patternsFileName = patternsPath.getName().toString();
                parseSkipFile(patternsFileName);
            }
            if (conf.getBoolean("wordcount.stop.words", true)) {
                Path stopWordsPath = new Path(URIs[1].getPath());
                String stopWordsFileName = stopWordsPath.getName().toString();
                parseStopWordsFile(stopWordsFileName);
            }

        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
            for (String pattern : patternsToSkip) {
                line = line.replaceAll(pattern, "");
            }
            StringTokenizer itr = new StringTokenizer(line);
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                if (token.length() < 3 || isNumber(token)) continue;
                for (String stopWord : stopWords) {
                    if (token.equals(stopWord)) continue;
                }
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

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();

        if (!(remainingArgs.length != 2 || remainingArgs.length != 4)) {
            System.err.println("Usage: wordcount <in> <out> [-skip skipPatternFile]");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(MapReduceWordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        List<String> otherArgs = new ArrayList<String>();
        for (int i = 0; i < remainingArgs.length; i++) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
            } else if ("-stop".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.stop.words", true);
            } else {
                otherArgs.add(remainingArgs[i]);
            }
        }

        FileInputFormat.addInputPath(job, new Path(otherArgs.get(0)));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs.get(1)));

        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }
}