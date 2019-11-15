using Random

struct Student
  name 
  score
end

N = 10000000
student_list = [Student(randstring(), rand()) for i in 1:N]

function lazy_student_gen(student_list)
  Channel() do c
    for st in student_list
      st.score < 0.7 && put!(c, st)
    end
  end
end

function student_filtered(student_list)
  st_lst = Student[]
  for st in student_list
    st.score < 0.7 && push!(st_lst, st)
  end
  return st_lst
end

@time cn = lazy_student_gen(student_list)
@time ls = student_filtered(student_list)
