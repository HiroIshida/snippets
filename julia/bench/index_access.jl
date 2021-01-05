struct Joint
    id::Int
end

function main()
    joints = Joint[]
    idxes = Int[]
    for i in 1:100
        push!(joints, Joint(i))
        push!(idxes, i)
    end

    ids = 0
    function bench1()
        for i in 1:10000
            for j in 1:100
                joint = joints[j]
                ids = joint.id
            end
        end
    end
    @time bench1()

    function bench2()
        for i in 1:10000
            for j in 1:100
                idx = idxes[j]
                joint = joints[idx]
                ids = joint.id
            end
        end
    end
    @time bench2()
end

main()
