#!/usr/bin/env ruby

if ARGV.length < 1
  raise "ARGV[0] should be the checkpoint dir"
elsif ARGV.length == 2
  MODEL = ARGV[1]
else
  MODEL = "distilbert"
end

validfile = "valid5000.csv"
cpdir = ARGV[0] + "/checkpoint*"

scores = Hash.new()
(Dir[cpdir] + [cpdir.split("/")[0..-2].join("/")]).sort.each { |d|
  puts(d)
  begin
    output = `python main.py -B 1 -e -d #{validfile} -l #{d} -a #{MODEL}`.split("\n")[-1]
    scores[d] = output
    puts(output)
    File.open(cpdir.split("/")[0..-2].join("/") + "/scores.txt", "w") { |f|
      scores.each { |k,v|
        begin
          f.puts(v + ": " + k)
        rescue
        end
      }
    }
  rescue
  end
}

File.open(cpdir.split("/")[0..-2].join("/") + "/scores.txt", "w") { |f|
  scores.each { |k,v|
    begin
      puts(v + ": " + k)
      f.puts(v + ": " + k)
    rescue
    end
  }
}
